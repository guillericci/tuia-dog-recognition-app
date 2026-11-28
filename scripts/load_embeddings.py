"""Carga en la base vectorial embeddings YA CALCULADOS de la galería (alternativa a build_index.py).

Los embeddings se calculan en 2-train.ipynb (sección 5/9) a partir de data/splits/train.csv y se guardan en un .npy
con el mismo orden de filas. Cada fila se guarda como lo hace SimilarityService.index_image: path relativo a
DATA_PATH, raza y metadata {"split", "model"}.

Desde el contenedor:
    docker compose exec backend python /scripts/load_embeddings.py --embeddings data/splits/baseline__train.npy --model baseline --reset
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if not (SRC / "lib").is_dir():
    # Dentro del contenedor del backend: src/ esta montado en /app (WORKDIR) y scripts/ en /scripts.
    ROOT = SRC = Path.cwd()
sys.path.insert(0, str(SRC))
os.chdir(SRC if (SRC / ".env").is_file() else ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, help=".npy con los embeddings, una fila por imagen del CSV.")
    parser.add_argument("--csv", default="data/splits/train.csv", help="CSV de la galería (path, breed).")
    parser.add_argument("--model", default=None, help="Modelo de los embeddings (por defecto, EMBEDDING_MODEL).")
    parser.add_argument("--split", default="train")
    parser.add_argument("--reset", action="store_true", help="Vacía la tabla antes de cargar (solo PostgreSQL).")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    from lib.bootstrap import build_store
    from lib.config import settings
    from lib.schemas import EmbeddingRecord

    emb = np.load(args.embeddings)
    df = pd.read_csv(args.csv)
    if len(emb) != len(df):
        sys.exit(f"{args.embeddings} tiene {len(emb)} filas y {args.csv} tiene {len(df)}: no corresponden.")
    if emb.shape[1] != settings.embedding_dim:
        sys.exit(f"Dimensión {emb.shape[1]} distinta de EMBEDDING_DIM={settings.embedding_dim}.")
    model = args.model or settings.embedding_model
    if not (settings.data_path / df["path"].iloc[0]).exists():
        print(f"Aviso: no encuentro {settings.data_path / df['path'].iloc[0]} (el dataset no está en este entorno).")

    store = build_store(settings)
    if hasattr(store, "conn"):                         # PostgreSQL + pgvector
        with store.conn.cursor() as cur:
            if args.reset:
                cur.execute("TRUNCATE embeddings")
                print("Tabla embeddings vaciada.")
            cur.execute("SELECT count(*) FROM embeddings")
            existentes = cur.fetchone()[0]
        if existentes:
            sys.exit(f"La tabla ya tiene {existentes} filas: usá --reset para reemplazarla (evita duplicados).")
    elif args.reset:
        sys.exit("--reset solo está soportado con PostgreSQL (USE_PGVECTOR=true).")

    for i, (path, breed) in enumerate(zip(df["path"], df["breed"]), start=1):
        store.append(EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=[float(x) for x in emb[i - 1]],
            path=path,
            breed=breed,
            metadata={"split": args.split, "model": model},
        ))
        if i % 1000 == 0:
            print(f"{i}/{len(df)}", flush=True)
    print(f"Total: {len(df)} embeddings almacenados (modelo: {model})")


if __name__ == "__main__":
    main()