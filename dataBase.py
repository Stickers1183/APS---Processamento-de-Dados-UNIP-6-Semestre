import biometricRecognition
import os
import pandas as pd
import glob
import numpy as np
import json

# --- Configuração ---
IMAGE_FOLDERS = ['handDatabase/working'] 
OUTPUT_CSV_FILE = 'measures.csv'
USERS_JSON_FILE = 'users.json'

# Threshold que funcionou
REGISTRATION_THRESHOLD = 0.20 

# Cargos especiais para os 3 primeiros usuários
SPECIAL_ROLES = [
    {"role": "Ministro", "access_level": 3},  # Para o primeiro usuário (ex: "01")
    {"role": "Diretor", "access_level": 2},   # Para o segundo usuário (ex: "02")
    {"role": "Publico", "access_level": 1}    # Para o terceiro usuário (ex: "03")
]

# Cargo padrão para qualquer usuário novo (ex: "04", "05"...)
DEFAULT_NEW_USER_ROLE = {
    "role": "Publico",
    "access_level": 1 
}
# --------------------

def main():
    print("--- Iniciando Geração de Banco de Dados 'Inteligente' ---")
    
    files = []
    extensions = ('*.jpg', '*.png', '*.jpeg')
    for folder in IMAGE_FOLDERS:
        if not os.path.isdir(folder):
            print(f"Aviso: A pasta '{folder}' não foi encontrada. Pulando...")
            continue
        print(f"Encontrando imagens em: {folder}")
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, ext)))
    
    if not files:
        print("Nenhum arquivo de imagem encontrado na pasta 'working'. Abortando.")
        return

    print(f"Total de {len(files)} imagens encontradas. Iniciando processamento...")

    all_data_list = []
    
    # --- Lógica de ID Numérico ---
    in_memory_db = [] # Lista de (features_np, numeric_person_id)
    
    # Mapeia o prefixo do arquivo (ex: "a") para o ID numérico (ex: "01")
    prefix_to_numeric_id_map = {} 
    
    # Mapeia o ID numérico (ex: "01") para o índice de cargo (ex: 0)
    user_id_to_index_map = {}
    
    next_user_id = 1      # Contador para o próximo ID (1, 2, 3...)
    current_user_index = 0 # Contador para o índice do cargo (0, 1, 2...)
    # ----------------------------------------

    for f_path in files:
        f_name = os.path.basename(f_path)
        print(f"\nProcessando: {f_name}")

        features, person_id_from_file, photo_id_from_file = biometricRecognition.processImage(f_path, output_dir=None)
        
        if features is None:
            print(f"...Falha ao processar a imagem {f_name}. Pulando.")
            continue
            
        features_np = np.array(features).astype(float)
        final_person_id = None # Este será o ID numérico (ex: "01")

        # 2. Lógica de "Cadastro Inteligente"
        if not in_memory_db:
            # É a primeira imagem de todas
            final_person_id = f"{next_user_id:02d}" # "01"
            print(f"...Este é o primeiro usuário. Registrando como: {final_person_id} (do arquivo {person_id_from_file})")
            
            prefix_to_numeric_id_map[person_id_from_file] = final_person_id
            user_id_to_index_map[final_person_id] = current_user_index
            current_user_index += 1
            next_user_id += 1
        else:
            # Compara com o banco de dados em memória
            min_dist = float('inf')
            match_person_id = None # ID numérico correspondente
            
            for db_features, db_person_id in in_memory_db:
                try:
                    dist_test = np.linalg.norm(features_np - db_features)
                    if dist_test < min_dist:
                        min_dist = dist_test
                        match_person_id = db_person_id
                except Exception as e:
                    print(f"Erro no cálculo da distância: {e}")
                    
            if min_dist < REGISTRATION_THRESHOLD:
                # CORRESPONDÊNCIA ENCONTRADA
                final_person_id = match_person_id
                print(f"...Correspondência encontrada. Imagem pertence ao usuário: {final_person_id} (Dist: {min_dist:.4f})")
                
                # Garante que o prefixo do arquivo (ex: "c") também aponte para este ID (ex: "01")
                if person_id_from_file not in prefix_to_numeric_id_map:
                    prefix_to_numeric_id_map[person_id_from_file] = final_person_id
            
            else:
                # NOVO USUÁRIO
                # Verifica se já vimos esse prefixo de arquivo (ex: "a") antes
                if person_id_from_file in prefix_to_numeric_id_map:
                    # Sim, já vimos "a". Mesmo que as feições sejam diferentes, agrupamos pelo prefixo
                    final_person_id = prefix_to_numeric_id_map[person_id_from_file]
                    print(f"...Sem correspondência biométrica (Dist: {min_dist:.4f}), mas o prefixo '{person_id_from_file}' já pertence ao usuário {final_person_id}.")
                else:
                    # É um prefixo de arquivo totalmente novo (ex: "b")
                    final_person_id = f"{next_user_id:02d}"
                    print(f"...Nenhuma correspondência. Registrando como NOVO usuário: {final_person_id} (do arquivo {person_id_from_file})")
                    
                    prefix_to_numeric_id_map[person_id_from_file] = final_person_id
                    user_id_to_index_map[final_person_id] = current_user_index
                    current_user_index += 1
                    next_user_id += 1

        # 4. Adiciona os dados processados na lista
        # O 'final_person_id' (ex: "01") é salvo no CSV
        new_row_data = features + [photo_id_from_file, final_person_id]
        all_data_list.append(new_row_data)
        
        # Adiciona ao DB em memória para as próximas comparações
        in_memory_db.append((features_np, final_person_id))

    # 5. Salva o banco de dados biométrico (CSV)
    if not all_data_list:
        print("\nNenhum dado biométrico foi processado. O 'measures.csv' não foi criado.")
        return
    
    try:
        final_df = pd.DataFrame(all_data_list, columns=biometricRecognition.CSV_COLUMN_NAMES)
        final_df.to_csv(OUTPUT_CSV_FILE, index=False)
        
        print(f"\n--- Geração 'Inteligente' Concluída ---")
        print(f"Banco de dados salvo em: {OUTPUT_CSV_FILE}")
        print("Usuários únicos registrados (agora com IDs numéricos):")
        print(final_df['person'].value_counts())
    except IOError as e:
        print(f"\nErro ao salvar BD: {e}")
        return

    # 6. Sincronizar 'users.json' com cargos especiais E o prefixo do arquivo
    try:
        print(f"\nSincronizando '{USERS_JSON_FILE}'...")
        
        users_data = {} # Começa um JSON limpo
        json_changed = False
        
        # Itera sobre o mapa de prefixo->ID para construir o JSON
        for prefix, numeric_id in prefix_to_numeric_id_map.items():
            if numeric_id in users_data: # Evita duplicatas se vários prefixos apontarem para o mesmo ID
                continue

            index = user_id_to_index_map[numeric_id] # Pega o índice de cargo (0, 1, 2...)
            
            if 0 <= index < len(SPECIAL_ROLES):
                new_role = SPECIAL_ROLES[index]
                role = new_role["role"]
                access_level = new_role["access_level"]
                print(f"...Atribuindo cargo especial: '{role}' (Nível {access_level}) para o ID {numeric_id}")
            else:
                new_role = DEFAULT_NEW_USER_ROLE
                role = new_role["role"]
                access_level = new_role["access_level"]
                print(f"...Atribuindo cargo padrão: '{role}' (Nível {access_level}) para o ID {numeric_id}")

            users_data[numeric_id] = {
                "role": role,
                "access_level": access_level,
                "file_prefix": prefix  # Salva o prefixo (ex: "a") para encontrar a imagem
            }
            json_changed = True

        if json_changed:
            with open(USERS_JSON_FILE, 'w', encoding='utf-8') as f:
                json.dump(users_data, f, indent=2, ensure_ascii=False)
            print(f"Sucesso! '{USERS_JSON_FILE}' foi (re)criado com os novos IDs numéricos e prefixos de arquivo.")
        else:
            print(f"Nenhum usuário novo encontrado. '{USERS_JSON_FILE}' já estava sincronizado.")

    except Exception as e:
        print(f"\nERRO CRÍTICO ao tentar atualizar '{USERS_JSON_FILE}': {e}")

if __name__ == "__main__":
    main()