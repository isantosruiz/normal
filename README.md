# App Python para Vercel: area bajo la curva normal

Aplicacion web en Flask que:
- recibe `mu` (media) y `sigma` (desviacion estandar) en cajas de texto (por omision `0` y `1`);
- recibe limites `DESDE` y `HASTA`;
- calcula el area bajo la curva normal en ese intervalo;
- dibuja la distribucion con el area sombreada;
- renderiza la expresion en LaTeX.
- permite limites infinitos con sintaxis tipo SymPy: `oo` y `-oo`.

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
3. Para produccion:
   ```bash
   vercel --prod
   ```
