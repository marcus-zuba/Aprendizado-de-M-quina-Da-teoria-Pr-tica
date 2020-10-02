import sys
sys.path.append("../")
from metodo_competicao import MetodoCompeticao
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from preprocessamento_atributos_competicao import gerar_atributos_ator, gerar_atributos_resumo
 
def gerar_saida_teste( df_data_to_predict, col_classe, num_grupo):
    """
    Assim como os demais códigos da pasta "competicao_am", esta função 
    só poderá ser modificada na fase de geração da solução. 
    """
    
    scikit_method = LinearSVC(C=2**1.027763, random_state=2)
    ml_method = MetodoCompeticao(scikit_method)
    
    df_treino = pd.read_csv("../datasets/movies_amostra.csv")
    
    #faz a predição baseada nos atores
    y_to_predict, arr_predictions_ator = ml_method.eval_actors(df_treino, df_data_to_predict, col_classe)
    #faz a predição baseada nos resumos
    y_to_predict, arr_predictions_bow = ml_method.eval_bow(df_treino, df_data_to_predict, col_classe)
    #faz a predição baseada nos titulos
    y_to_predict, arr_predictions_titulo = ml_method.eval_titulo(df_treino, df_data_to_predict, col_classe)
            
    #combina as tres representações por maioria
    arr_final_predictions = ml_method.combine_predictions(arr_predictions_ator, arr_predictions_bow, arr_predictions_titulo)
    
    #grava o resultado obtido
    with open(f"predict_grupo_{num_grupo}.txt","w") as file_predict:
        for predict in arr_final_predictions:
            file_predict.write(ml_method.dic_int_to_nom_classe[predict]+"\n")
            