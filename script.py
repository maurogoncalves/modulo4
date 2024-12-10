from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_classification(y_true, y_pred):
    """
    Avalia um modelo de classificação com base nas métricas: acurácia, sensibilidade, especificidade,
    precisão e F-score.

    :param y_true: Lista ou array com os rótulos reais.
    :param y_pred: Lista ou array com os rótulos previstos pelo modelo.
    :return: Um dicionário contendo as métricas calculadas.
    """
    # Calcula a matriz de confusão
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Cálculo das métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensibilidade
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f_score = f1_score(y_true, y_pred, zero_division=0)

    # Resultados
    metrics = {
        "Acurácia": accuracy,
        "Sensibilidade (Recall)": recall,
        "Especificidade": specificity,
        "Precisão": precision,
        "F-score": f_score
    }
    
    return metrics

# Exemplo de uso
if __name__ == "__main__":
    # Rótulos reais e previstos (exemplo)
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]

    # Avaliação do modelo
    resultados = evaluate_classification(y_true, y_pred)
    
    # Exibição dos resultados
    for metric, value in resultados.items():
        print(f"{metric}: {value:.2f}")
