import pandas as pd
import numpy as np
import joblib
from statsmodels.api import add_constant

# Carregar o modelo treinado e o scaler
model_path = 'final_model.pkl'
scaler_path = 'scaler.pkl'

final_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Função para normalizar os dados de entrada
def normalize_input(user_input, scaler):
    """
    Normaliza os dados do usuário com base no scaler usado no treinamento.
    """
    cols_to_normalize = ['Age', 'Avg_BPM', 'Resting_BPM', 'Session_Duration', 'Calories_Burned']
    normalized_values = scaler.transform([[user_input[col] for col in cols_to_normalize]])
    normalized_data = dict(zip([col + '_N' for col in cols_to_normalize], normalized_values[0]))
    normalized_data.update({
        'Gender': user_input['Gender'],
        'Workout_Type': user_input['Workout_Type']
    })
    return normalized_data

# Função para desnormalizar os valores normalizados
def denormalize_value(normalized_value, mean, std):
    """
    Desnormaliza um valor com base na média e no desvio padrão.
    """
    return (normalized_value * std) + mean

# Função para gerar sugestões de treino
def suggest_training(user_data, target_calories=None):
    """
    Sugere ajustes no treino com base no modelo treinado e normaliza/desnormaliza corretamente.
    """

    if target_calories:
      # Incluir a variável alvo para normalização
      user_data['Calories_Burned'] = target_calories
    else:
      # Atribui valor padrão se não for fornecido
      user_data['Calories_Burned'] = 0

    # Normalizar os dados de entrada
    user_data_normalized = normalize_input(user_data, scaler)

    # Preparar os dados para o modelo
    user_df = pd.DataFrame([user_data_normalized])
    user_df = add_constant(user_df, has_constant='add')

    # Fazer a previsão (resultado normalizado)
    predicted_calories_normalized = final_model.predict(user_df)[0]

    # Desnormalizar as calorias previstas
    calories_mean = scaler.mean_[scaler.feature_names_in_.tolist().index('Calories_Burned')]  # Última variável no scaler é `Calories_Burned`
    calories_std = scaler.scale_[scaler.feature_names_in_.tolist().index('Calories_Burned')]
    predicted_calories = denormalize_value(predicted_calories_normalized, calories_mean, calories_std)

    # Normalizar a meta de calorias
    if target_calories:
        target_calories_normalized = user_data_normalized['Calories_Burned_N']
    else:
        target_calories_normalized = None

    print(f"Duração da sessão: {user_data['Session_Duration']} horas")

    # Gerar sugestão com base na meta
    suggestion = {"Calories_Predicted": f"{predicted_calories:.2f}"}

    if target_calories_normalized:
        # Ajustar a duração para atingir a meta
        current_duration_normalized = user_data_normalized['Session_Duration_N']
        adjustment_factor = target_calories_normalized / predicted_calories_normalized
        suggested_duration_normalized = current_duration_normalized * adjustment_factor

        print(f"Meta de calorias normalizada: {target_calories_normalized:.2f}")
        print(f"Fator de ajuste: {adjustment_factor:.2f}")
        print(f"Duração Sugerida Normalizada: {suggested_duration_normalized:.2f}")

        # Desnormalizar a duração sugerida
        duration_mean = scaler.mean_[scaler.feature_names_in_.tolist().index('Session_Duration')]
        duration_std = scaler.scale_[scaler.feature_names_in_.tolist().index('Session_Duration')]
        suggested_duration = denormalize_value(suggested_duration_normalized, duration_mean, duration_std)

        suggestion["Suggested_Duration"] = f"{suggested_duration:.2f}"
        suggestion["Message"] = (
            f"Para atingir {target_calories:.2f} calorias, ajuste a duração do treino para {suggested_duration:.2f} horas."
        )
    else:
        suggestion["Message"] = (
            f"Com as informações fornecidas, você queimará aproximadamente {predicted_calories:.2f} calorias."
        )

    return suggestion

def show_suggestion(suggestion):
  for key, value in suggestion.items():
    print(f"{key}: {value}")

# Dicionário com dados de exemplo
user_input = {
    "Gender": 1,              # 0: Feminino, 1: Masculino
    "Workout_Type": 0,        # Tipo de treino
    "Age": 30,                # Idade
    "Avg_BPM": 140,           # BPM médio
    "Resting_BPM": 75,        # BPM em repouso
    "Session_Duration": 1.5   # Duração da sessão (horas)
}

# Meta de calorias fornecida de forma real
target_calories = 1000  # Calorias reais

# Criando um DataFrame a partir do dicionário
df_user_input = pd.DataFrame([user_input])

# Gerando 10 linhas com dados aleatórios semelhantes ao exemplo
num_rows = 10
df_test = pd.DataFrame(index=np.arange(num_rows), columns=df_user_input.columns)

for col in df_user_input.columns:
  if col == 'Gender':
    # Valores binários para gênero
    df_test[col] = np.random.randint(0, 2, num_rows)
  elif col == 'Workout_Type':
    # Tipos de treinos (pode ser ajustado)
    df_test[col] = np.random.randint(0, 4, num_rows)
  elif col == 'Age':
    df_test[col] = np.random.randint(18, 60, num_rows)
  elif col == 'Avg_BPM':
    df_test[col] = np.random.randint(120, 180, num_rows)
  elif col == 'Resting_BPM':
    df_test[col] = np.random.randint(60, 80, num_rows)
  elif col == 'Session_Duration':
    df_test[col] = np.round(np.random.uniform(0.5, 2.5, num_rows), decimals=2)

for test_input in df_test.to_dict(orient='records'):
    # Aplicar o modelo para os dados de teste
    suggestion = suggest_training(test_input, target_calories)

    # Exibir resultados
    show_suggestion(suggestion)
    print("\n")