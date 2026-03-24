import base64
import io
import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np
import sympy as sp
from flask import Flask, render_template, request

matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = Flask(__name__)

DEFAULT_VALUES = {
    "mu": "0",
    "sigma": "1",
    "desde": "-1",
    "hasta": "1",
}

POS_INF_TOKENS = {"oo", "+oo", "inf", "+inf", "infinity", "+infinity", "∞", "+∞"}
NEG_INF_TOKENS = {"-oo", "-inf", "-infinity", "-∞"}


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coef * np.exp(exponent)


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    scaled = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(scaled))


def format_number(value: float) -> str:
    normalized = 0.0 if abs(value) < 1e-12 else value
    return f"{normalized:.6g}"


def parse_finite_value(raw_value: str, field_name: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Ingresa un valor numérico válido para {field_name}.") from exc

    if not math.isfinite(value):
        raise ValueError(f"{field_name} debe ser finito.")

    return value


def parse_bound(raw_value: str) -> float:
    token = raw_value.strip().lower().replace(" ", "")
    if token in POS_INF_TOKENS:
        return math.inf
    if token in NEG_INF_TOKENS:
        return -math.inf

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError("Ingresa límites válidos. Usa números o oo/-oo.") from exc


def format_bound_latex(value: float) -> str:
    if math.isinf(value):
        return r"\infty" if value > 0 else r"-\infty"
    return format_number(value)


def format_bound_plot(value: float) -> str:
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    return format_number(value)


def to_sympy_number(value: float) -> sp.Expr:
    if abs(value - round(value)) < 1e-12:
        return sp.Integer(int(round(value)))
    return sp.Float(value)


def build_density_latex(mu: float, sigma: float) -> str:
    z = sp.Symbol("z", real=True)
    mu_expr = to_sympy_number(mu)
    sigma_expr = to_sympy_number(sigma)

    coefficient = sp.simplify(1 / (sigma_expr * sp.sqrt(2 * sp.pi)))
    exponent = sp.simplify(-((z - mu_expr) ** 2) / (2 * sigma_expr**2))

    coefficient_latex = sp.latex(coefficient)
    exponent_latex = sp.latex(exponent)

    if coefficient == 1:
        return rf"\exp\left({exponent_latex}\right)"

    return rf"{coefficient_latex}\,\exp\left({exponent_latex}\right)"


def draw_curve(mu: float, sigma: float, desde: float, hasta: float) -> str:
    left, right = sorted((desde, hasta))
    candidates = [mu - 4.0 * sigma, mu + 4.0 * sigma]
    if math.isfinite(left):
        candidates.extend([left - sigma, left + sigma])
    if math.isfinite(right):
        candidates.extend([right - sigma, right + sigma])

    x_min = min(candidates)
    x_max = max(candidates)

    x = np.linspace(x_min, x_max, 1400)
    y = normal_pdf(x, mu, sigma)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, y, color="#005f73", linewidth=2.2, label="g(z)")

    if math.isinf(left) and math.isinf(right):
        mask = np.ones_like(x, dtype=bool)
    elif math.isinf(left):
        mask = x <= right
    elif math.isinf(right):
        mask = x >= left
    else:
        mask = (x >= left) & (x <= right)
    left_label = format_bound_plot(left)
    right_label = format_bound_plot(right)
    area_label = f"P({left_label} < z < {right_label})"
    ax.fill_between(x[mask], 0, y[mask], color="#ee9b00", alpha=0.35, label=area_label)

    if math.isfinite(left):
        ax.axvline(left, color="#ae2012", linestyle="--", linewidth=1.2)
    if math.isfinite(right):
        ax.axvline(right, color="#ae2012", linestyle="--", linewidth=1.2)
    ax.set_title("Distribución normal y área bajo la curva")
    ax.set_xlabel("z")
    ax.set_ylabel("Densidad")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right")

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def build_latex(mu: float, sigma: float, desde: float, hasta: float, area: float) -> tuple[str, str]:
    left, right = sorted((desde, hasta))
    left_s = format_bound_latex(left)
    right_s = format_bound_latex(right)
    density_latex = build_density_latex(mu, sigma)

    latex_density = rf"g(z):={density_latex}"
    latex_probability = (
        rf"P\left({left_s} < z < {right_s}\right)="
        rf"\int_{{{left_s}}}^{{{right_s}}}g(z)\,dz={area:.6f}"
    )
    return latex_density, latex_probability


@app.route("/", methods=["GET", "POST"])
def index():
    form_values = dict(DEFAULT_VALUES)
    error_message = None
    result = None

    if request.method == "POST":
        form_values = {
            "mu": (request.form.get("mu") or DEFAULT_VALUES["mu"]).strip(),
            "sigma": (request.form.get("sigma") or DEFAULT_VALUES["sigma"]).strip(),
            "desde": (request.form.get("desde") or DEFAULT_VALUES["desde"]).strip(),
            "hasta": (request.form.get("hasta") or DEFAULT_VALUES["hasta"]).strip(),
        }

        try:
            mu = parse_finite_value(form_values["mu"], "la media")
            sigma = parse_finite_value(form_values["sigma"], "la desviación estándar")
            desde = parse_bound(form_values["desde"])
            hasta = parse_bound(form_values["hasta"])

            if sigma <= 0:
                raise ValueError("La desviación estándar debe ser mayor que 0.")

            left, right = sorted((desde, hasta))
            area = normal_cdf(right, mu, sigma) - normal_cdf(left, mu, sigma)
            image_b64 = draw_curve(mu, sigma, left, right)
            latex_density, latex_probability = build_latex(mu, sigma, left, right, area)

            result = {
                "area": f"{area:.6f}",
                "image_b64": image_b64,
                "latex_density": latex_density,
                "latex_probability": latex_probability,
            }
        except ValueError as exc:
            error_message = str(exc)

    return render_template(
        "index.html",
        form_values=form_values,
        error_message=error_message,
        result=result,
    )


if __name__ == "__main__":
    app.run(debug=True)
