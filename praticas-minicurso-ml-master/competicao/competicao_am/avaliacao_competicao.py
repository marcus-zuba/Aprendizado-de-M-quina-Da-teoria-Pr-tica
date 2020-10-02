import sys
sys.path.append("../")
from base_am.avaliacao import OtimizacaoObjetivo
from base_am.metodo import MetodoAprendizadoDeMaquina
from base_am.resultado import Fold, Resultado
from base_am.avaliacao import Experimento
from competicao_am.metodo_competicao import MetodoCompeticao
import optuna
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class OtimizacaoObjetivoSVMCompeticao(OtimizacaoObjetivo):
    def __init__(self, fold:Fold, num_arvores_max:int=5):
        super().__init__(fold)
        self.num_arvores_max = num_arvores_max

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        #Um custo adequado para custo pode variar muito, por ex, para uma tarefa 
        #o valor de custo pode ser 10, para outra, 32000. 
        #Assim, normalmente, para conseguir valores mais distintos,
        #usamos c=2^exp_cost
        exp_cost = trial.suggest_uniform('min_samples_split', 0, 7) 

        scikit_method = LinearSVC(C=2**exp_cost, random_state=2)

        return MetodoCompeticao(scikit_method)

    def resultado_metrica_otimizacao(self,resultado: Resultado) -> float:
        return resultado.macro_f1
    

class OtimizacaoObjetivoRandomForest(OtimizacaoObjetivo):
    def init(self, fold:Fold):
        super().init(fold)

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        min_samples_split = trial.suggest_uniform('min_samples_split', 0, 0.5)
        max_features = trial.suggest_uniform('max_features', 0, 0.5)
        num_arvores = trial.suggest_int('num_arvores', 1, 5)
        clf_rf = RandomForestClassifier(n_estimators=num_arvores, min_samples_split=min_samples_split, max_features=max_features, random_state=2)

        return MetodoCompeticao(clf_rf)

    def resultado_metrica_otimizacao(self, resultado:Resultado) ->float:
        return resultado.macro_f1    

    
class OtimizacaoObjetivoArvoreDecisao(OtimizacaoObjetivo):
    def init(self, fold:Fold):
        super().init(fold)

    def obtem_metodo(self,trial: optuna.Trial) -> MetodoAprendizadoDeMaquina:

        min_samples = trial.suggest_uniform('min_samples_split', 0, 0.5)
        clf_dtree = DecisionTreeClassifier(min_samples_split=min_samples,random_state=2)

        return MetodoCompeticao(clf_dtree)

    def resultado_metrica_otimizacao(self,resultado):
        return resultado.macro_f1
    
    
def executar_experimentos():
    
    df_treino = pd.read_csv("../datasets/movies_amostra.csv")
    
    folds = Fold.gerar_k_folds(df_treino, val_k=5, col_classe="genero", num_repeticoes=1, 
                               seed=1, num_folds_validacao=3, num_repeticoes_validacao=2)

    
    SVM = LinearSVC(random_state=2)
    RandomForest = RandomForestClassifier(random_state=2)
    DecisionTree = DecisionTreeClassifier(random_state=2)
    
    ml_method_SVM = MetodoCompeticao(SVM)
    ml_method_RF = MetodoCompeticao(RandomForest)
    ml_method_DT = MetodoCompeticao(DecisionTree)
    
    exp_SVM = Experimento(folds, ml_method_SVM, OtimizacaoObjetivoSVMCompeticao, num_trials=100)
    exp_RF = Experimento(folds, ml_method_SVM, OtimizacaoObjetivoRandomForest, num_trials=100)
    exp_DT = Experimento(folds, ml_method_SVM, OtimizacaoObjetivoArvoreDecisao, num_trials=100)
    
    
    resultados_SVM = exp_SVM.calcula_resultados()
    resultados_RF = exp_RF.calcula_resultados()
    resultados_DT = exp_DT.calcula_resultados()    
    
    #PRINTAR MÉTRICAS DOS EXPERIMENTOS DE CADA MODELO
    #PARA DEFINIR QUAL É O MELHOR
        
    print("RESULTADOS SVM")
    
    for resultado in resultados_SVM:
        print("Matriz de Confusao:")
        print(resultado.mat_confusao)
        print(f'Revocacao: {resultado.revocacao}')
        print(f'Precisao: {resultado.precisao}')
        print(f'Acuracia: {resultado.acuracia}')
        print(f'MacroF1: {resultado.macro_f1}')
        print("\n")
    
    print("\n\nRESULTADOS RF")

    for resultado in resultados_RF:
        print("Matriz de Confusao:")
        print(resultado.mat_confusao)
        print(f'Revocacao: {resultado.revocacao}')
        print(f'Precisao: {resultado.precisao}')
        print(f'Acuracia: {resultado.acuracia}')
        print(f'MacroF1: {resultado.macro_f1}')
        print("\n")
        
    print("\n\nRESULTADOS DT")
   
    for resultado in resultados_DT:
        print("Matriz de Confusao:")
        print(resultado.mat_confusao)
        print(f'Revocacao: {resultado.revocacao}')
        print(f'Precisao: {resultado.precisao}')
        print(f'Acuracia: {resultado.acuracia}')
        print(f'MacroF1: {resultado.macro_f1}')
        print("\n")
        

    #RETORNA UM VETOR COM OS TRÊS EXPERIMENTOS
    #MAS VAMOS UTILIZAR SÓ O EXPERIMENTO DO MELHOR MODELO
        
    experimentos = []
    #experimento 0 - SVM
    experimentos.append(exp_SVM)
    #experimento 1 - RF
    experimentos.append(exp_RF)    
    #experimento 2 - DT
    experimentos.append(exp_DT)    
        
    return experimentos


def testar_parametros(df_treino, df_data_to_predict, min_samples_split):
    
    scikit_method = LinearSVC(C=2**(min_samples_split), random_state=2)
    ml_method = MetodoCompeticao(scikit_method)
    
    return ml_method.eval(df_treino, df_data_to_predict, "genero")

