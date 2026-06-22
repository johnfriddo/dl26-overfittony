import os
import argparse
import pandas as pd
import numpy as np


def timestamp_to_seconds(ts_series):
    ts_series = ts_series.astype(str).str.strip()
    seconds = []
    for ts in ts_series:
        try:
            parts = ts.split(':')
            if len(parts) == 3:
                h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                seconds.append(h * 3600 + m * 60 + s)
            elif len(parts) == 2:
                m, s = float(parts[0]), float(parts[1])
                seconds.append(m * 60 + s)
            else:
                seconds.append(float(parts[0]))
        except Exception:
            seconds.append(np.nan)
    return pd.Series(seconds)


def build_multimodal_train(sounds_train_path, kitchens_train_path, output_path, threshold=0.5):
    """
    Costruisce multimodal_train.csv joinando EPIC-Sounds train con EPIC-Kitchens train.

    Il join avviene SOLO tra i rispettivi split di train: nessun evento audio
    del val/test può entrare qui perché sounds_train_path contiene solo il train
    di EPIC-Sounds. Il no-leakage è garantito per costruzione dagli input.

    Metrica di overlap: containment ratio = intersezione / durata_audio.
    Misura quanta parte dell'evento sonoro cade dentro l'azione video.
    Preferita alla tIoU classica perché quest'ultima penalizza gli eventi
    sonori brevi contenuti in azioni lunghe (caso dominante in EPIC-Sounds).
    """
    print("Caricamento EPIC-Sounds train...")
    df_sounds = pd.read_csv(sounds_train_path)
    df_sounds['video_id'] = df_sounds['video_id'].astype(str).str.strip()
    df_sounds['start_sec'] = timestamp_to_seconds(df_sounds['start_timestamp'])
    df_sounds['stop_sec'] = timestamp_to_seconds(df_sounds['stop_timestamp'])
    df_sounds = df_sounds.dropna(subset=['start_sec', 'stop_sec'])

    print("Caricamento EPIC-Kitchens train...")
    df_kitchens = pd.read_csv(kitchens_train_path)
    df_kitchens['video_id'] = df_kitchens['video_id'].astype(str).str.strip()
    df_kitchens['start_sec'] = timestamp_to_seconds(df_kitchens['start_timestamp'])
    df_kitchens['stop_sec'] = timestamp_to_seconds(df_kitchens['stop_timestamp'])
    df_kitchens = df_kitchens.dropna(subset=['start_sec', 'stop_sec'])

    # Cross-join per video_id: ogni evento sonoro viene confrontato con tutte
    # le azioni video dello stesso video.
    print("Cross-join per video_id...")
    df_merged = pd.merge(df_kitchens, df_sounds, on='video_id', suffixes=('_video', '_audio'))

    print("Calcolo containment ratio...")
    max_start = np.maximum(df_merged['start_sec_video'], df_merged['start_sec_audio'])
    min_stop = np.minimum(df_merged['stop_sec_video'], df_merged['stop_sec_audio'])
    intersection = np.maximum(0.0, min_stop - max_start)
    audio_duration = (df_merged['stop_sec_audio'] - df_merged['start_sec_audio']).clip(lower=1e-8)
    df_merged['containment_ratio'] = intersection / audio_duration

    df_joined = df_merged[df_merged['containment_ratio'] >= threshold].copy()
    print(f"Coppie con containment >= {threshold}: {len(df_joined)}")

    # Deduplicazione: un evento sonoro può matchare più azioni consecutive.
    # Teniamo solo il match con containment più alto (best-match per annotation_id).
    df_joined = df_joined.sort_values('containment_ratio', ascending=False)
    dup_key = 'annotation_id_audio' if 'annotation_id_audio' in df_joined.columns else 'annotation_id'
    df_joined = df_joined.drop_duplicates(subset=[dup_key])
    print(f"Dopo deduplicazione (best-match per evento sonoro): {len(df_joined)}")

    # Alias colonne ambigue per il dataloader
    df_joined['start_frame'] = df_joined['start_frame_video'] if 'start_frame_video' in df_joined.columns else df_joined['start_frame']
    df_joined['stop_frame'] = df_joined['stop_frame_video'] if 'stop_frame_video' in df_joined.columns else df_joined['stop_frame']
    df_joined['start_sample'] = df_joined['start_sample_audio'] if 'start_sample_audio' in df_joined.columns else df_joined['start_sample']
    df_joined['stop_sample'] = df_joined['stop_sample_audio'] if 'stop_sample_audio' in df_joined.columns else df_joined['stop_sample']

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_joined.to_csv(output_path, index=False)
    print(f"Salvato: {output_path} ({len(df_joined)} righe)")
    return df_joined


def build_audio_only_split(sounds_path, output_path):
    """
    Copia il CSV audio as-is senza alcun join con EPIC-Kitchens.
    Val e test vengono valutati su audio puro — identico al baseline AST.
    """
    df = pd.read_csv(sounds_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Salvato: {output_path} ({len(df)} righe)")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Costruisce i CSV multimodali per la distillazione student.")
    parser.add_argument("--sounds_train", type=str,
                        default="/home/rsnnng02c19b202w/dl26-overfittony/data/epic-sounds-annotations/EPIC_Sounds_train.csv")
    parser.add_argument("--sounds_val", type=str,
                        default="/home/rsnnng02c19b202w/dl26-overfittony/data/epic-sounds-annotations/EPIC_Sounds_validation.csv")
    parser.add_argument("--kitchens_train", type=str,
                        default="/home/gnfmrc01b01a494o/dataset/EPIC_100_train.csv")
    parser.add_argument("--output_dir", type=str,
                        default="/home/rsnnng02c19b202w/dl26-overfittony/dataset")
    parser.add_argument("--containment_threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_multimodal_train(
        sounds_train_path=args.sounds_train,
        kitchens_train_path=args.kitchens_train,
        output_path=os.path.join(args.output_dir, "multimodal_train.csv"),
        threshold=args.containment_threshold,
    )

    build_audio_only_split(
        sounds_path=args.sounds_val,
        output_path=os.path.join(args.output_dir, "multimodal_val.csv"),
    )

    print("Pipeline annotazioni completata.")
