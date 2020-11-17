from flask import Flask, render_template, request, jsonify
from treinamento_e_tratamento_modelo_aprendizado import trata_dataset
import numpy as np
import joblib


"""
0. Número de vezes em que ficou grávida.
1. Concentração de glicose.
2. Pressão diastólica (mm Hg).
3. Espessura da dobra cutânea do tríceps (mm).
4. Insulina (mu U/ml).
5. Índice de massa corporal (peso em kg/(altura em m)^2).
6. Histórico familiar de diabetes.
7. Idade (anos).
8. Classificação (0 ou 1 - 0 não diabético / 1 diabético ).
"""
campos = ["gest", "glic", "pressDiast", "espessura", "insulina", "imc", "heranca", "age"]

def converter_para_tipo_do_elemento(x):
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x

def gera_lista_formulario(form):
    lista_valores_formulario = []
    for campo in campos:
        print(form[campo])
        lista_valores_formulario.append(converter_para_tipo_do_elemento(form[campo]))
    print(lista_valores_formulario)
    return lista_valores_formulario

def previsao_diabetes(lista_valores_formulario):
    prever = np.array(lista_valores_formulario).reshape(1, 8)
    modelo_salvo = joblib.load("melhor_modelo.sav")
    resultado = modelo_salvo.predict(prever)
    return resultado[0]

app = Flask(__name__)
@app.route('/', methods=['GET'])
def home():
    return render_template("Previsao.html"), 200


@app.route('/result', methods=['POST'])
def result():
    formulario = request.form
    lista_valores_formulario = gera_lista_formulario(formulario)
    resultado = previsao_diabetes(lista_valores_formulario)
    if resultado:
       previsao="Em nossa previsao o paciente possui Diabetes {}".format(resultado)
    else:
       previsao="Em nossa previsao o paciente não possui Diabetes".format(resultado)
    return render_template("Previsao.html", previsao=previsao)


app.run(port=5000,debug=True)