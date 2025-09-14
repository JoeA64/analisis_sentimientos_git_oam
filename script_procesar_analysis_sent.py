#import pyodbc

import os

# Forzar todas las librerías a usar /tmp en lugar de /.cache (esto era para usar en glue)
#os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
#os.environ["HF_HOME"] = "/tmp/hf_cache"
#os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache"
#os.environ["XDG_CACHE_HOME"] = "/tmp/hf_cache"   # <- clave para evitar /.cache


# Librerías
import pandas as pd
import boto3
import io
from transformers import pipeline
import torch
import re

# Configuración S3
s3_bucket = 'sent-analysis-pr-aws'            # Reemplaza con tu bucket
s3_input_key = 'sql_export/respuestas_encuesta_2907_oam_sql3.csv'  # Ruta dentro del bucket
s3_output_key = "sql_export/respuestas_encuesta_2907_oam_resultado.parquet"


aws_region = 'us-east-2'

s3_client = boto3.client(
    's3',
    #aws_access_key_id=aws_access_key_id,
    #aws_secret_access_key=aws_secret_access_key,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=aws_region
)

# ️Leer CSV desde S3
response = s3_client.get_object(Bucket=s3_bucket, Key=s3_input_key)
df_encuesta = pd.read_parquet(io.BytesIO(response['Body'].read()))
#df_encuesta.head()

# 4️⃣ Preparar modelo HuggingFace
classifier_5cat = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def limpiar_texto(texto):
    if texto is None:
        return ""
    texto = str(texto).strip()
    texto = re.sub(r'\s+', ' ', texto)  # reemplaza múltiples espacios/saltos por uno solo
    return texto

# Limpieza primero
df_encuesta['respuesta_limpia'] = df_encuesta['RESPUESTA'].apply(limpiar_texto)

# Cortar textos largos a 512 caracteres
#df_encuesta['respuesta_limpia'] = df_encuesta['RESPUESTA'].astype(str).str.slice(0, 512)

# Procesamiento por lotes
from tqdm import tqdm
def analizar_sentimientos_batch(textos, batch_size=8):
    resultados = []
    for i in tqdm(range(0, len(textos), batch_size), desc="Procesando lotes"):
        batch = textos[i:i+batch_size].tolist()
        predicciones = classifier_5cat(batch)
        resultados.extend(predicciones)
    return resultados

# Ejecutar análisis
resultados = analizar_sentimientos_batch(df_encuesta['respuesta_limpia'], batch_size=8)

# Convertir la lista de resultados a DataFrame
df_resultados = pd.DataFrame(resultados)

# Renombrar columnas según tu modelo
df_resultados.columns = ['sentimiento_label', 'sentimiento_score']

# Concatenar con el DataFrame original
df_enriquecido2 = pd.concat([df_encuesta.reset_index(drop=True), df_resultados], axis=1)

# df_enriquecido2 es el DataFrame con los resultados del análisis

# Crear un buffer en memoria
#csv_buffer = io.StringIO()
parquet_buffer = io.BytesIO()
#df_enriquecido2.to_csv(csv_buffer, index=False)
df_enriquecido2.to_parquet(parquet_buffer, index=False, engine="pyarrow")

# Configuración del cliente S3
s3 = boto3.client(
    "s3",
    region_name="us-east-2",  # Ajusta a tu región
   
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

bucket_name = "sent-analysis-pr-aws"
output_key = "sql_export/respuestas_encuesta_2907_oam_resultado.parquet"

# Subir el CSV a S3
s3.put_object(
    Bucket=bucket_name,
    Key=output_key,
    Body=parquet_buffer.getvalue()
)

print(f"Archivo subido correctamente a s3://{bucket_name}/{output_key}")

