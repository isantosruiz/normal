# App Python para Vercel: área bajo la curva normal

Aplicación web en Flask que:
- recibe `mu` (media) y `sigma` (desviación estándar) en cajas de texto (por omisión `0` y `1`);
- recibe límites `DESDE` y `HASTA`;
- calcula el área bajo la curva normal en ese intervalo;
- dibuja la distribución con el área sombreada;
- renderiza la expresión en LaTeX;
- permite límites infinitos con sintaxis tipo SymPy: `oo` y `-oo`.

## Ejecutar en local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 api/index.py
```

Luego abre `http://127.0.0.1:5000`.

Si aparece un error al compilar `matplotlib`, elimina el entorno y reinstala para forzar versiones con wheel:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python3 api/index.py
```

## Desplegar en Vercel

1. Instala la CLI:
   ```bash
   npm i -g vercel
   ```
2. En la carpeta del proyecto, ejecuta:
   ```bash
   vercel
   ```
3. Para producción:
   ```bash
   vercel --prod
   ```
