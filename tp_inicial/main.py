import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from time import gmtime,strftime
from sagemaker.predictor import csv_serializer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

############Seteo variables iniciales Sagemaker y S3 
role = get_execution_role()
prefix = 'data'
data_file_name = 'input_scoring_data.csv'
bucket_name = 's3-labo-grupo-14-scoring-crediticio'
my_region = boto3.session.Session().region_name
s3 = boto3.resource('s3')

############Lectura del CSV
df_read = pd.read_csv('s3://{}/{}/{}'.format(bucket_name,prefix,data_file_name))
df_read

############Se quitan columnas del dataset que no van a usarse
df_sanitized = df_read.copy()
df_sanitized = df_sanitized[pd.notnull(df_sanitized['Num_of_Delayed_Payment'])]
X = df_sanitized.drop(columns=['Credit_Score','ID','Month','Name','Payment_Behaviour'],axis=1)
y = df_sanitized['Credit_Score']

############Las etiquetas de las columnas se pasan a índices numéricos
le = LabelEncoder()
df = X.copy()
for i in df:
    if df[i].dtype == 'object':
        df[i] = le.fit_transform(df[i])
    else:
        continue
        
df.info()

############Se transforman los valores de las columnas a float
X = np.array(X).astype('float32')
y = np.array(y).astype('float32')
##np.set_printoptions(threshold=np.inf)

############Se separa la data tanto para entrenamiento como para testeo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

import io
import sagemaker.amazon.common as smac

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf,X_train,y_train)
buf.seek(0)

############Se arma la estructura de carpetas donde va a ir tanto la data de entrenamiento/testing como los artefactos del modelo entrenado con Sagemaker
bucket = bucket_name
key  = 'linear-train-data'
boto3.resource('s3').Bucket(bucket).Object(
    os.path.join(prefix, 'train', key)).upload_fileobj(buf)
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('Uploaded training data location {}'.format(s3_train_data))

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf,X_test,y_test)
buf.seek(0)

key  = 'linear-test-data'
boto3.resource('s3').Bucket(bucket).Object(
    os.path.join(prefix, 'test', key)).upload_fileobj(buf)
s3_test_data = 's3://{}/{}/test/{}'.format(bucket, prefix, key)
print('Uploaded training data location {}'.format(s3_test_data))

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('Training artifacts will be uploaded to {}'.format(output_location))

############Configuración del algoritmo a utilizar de Sagemaker
role = get_execution_role()
region = boto3.Session().region_name
X_train.shape
sess = sagemaker.Session()
container = sagemaker.image_uris.retrieve("linear-learner",region)
linear = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count = 1,
    instance_type = 'ml.m4.xlarge',
    output_path = output_location,
    sagemaker_session = sess
)

linear.set_hyperparameters(
    num_classes = 3,
    predictor_type = 'multiclass_classifier',
    mini_batch_size = 14
)

df.info

############Entrenamiento del modelo
linear.fit({'train': s3_train_data, 'validation': s3_test_data})

############Despliegue del modelo
linear_predictor = linear.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')

############Predicciones del modelo entrenado
from sagemaker.predictor import csv_serializer, json_deserializer
linear_predictor.serializer = csv_serializer
linear_predictor.deserializer = json_deserializer

result = linear_predictor.predict(X_test)
result

predictions = np.array([r['score'] for r in result['predictions']])
predictions

############Limpieza endpoints y otros recursos usados en Sagemaker
sagemaker.Session().delete_endpoint(linear_predictor.endpoint_name)
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()