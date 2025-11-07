import os
import json
import glob
import pandas as pd
from flask import Flask, render_template_string, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.utils import secure_filename 

import biometricRecognition 
import dataBase 

# --- Configuração da Aplicação ---
app = Flask(__name__)
app.secret_key = 'aps-unip-pivc-2025' 

# Definindo todas as pastas
UPLOAD_FOLDER = 'handDatabase/working'
NOT_WORKING_FOLDER = 'handDatabase/notWorking'
TEMP_FOLDER = 'temp_uploads'
ENTRY_TEST_FOLDER = 'handDatabase/entryTest' # MUDANÇA: Usado APENAS para UPLOAD/TRIAGEM

# Threshold que funcionou
AUTH_THRESHOLD = 0.20 
# --------------------

# --- Carregamento dos Bancos de Dados na Inicialização ---
DB_USERS = {}
DB_BIOMETRIC = pd.DataFrame() # Inicia vazio

def load_permissions():
    """ Carrega o 'users.json' (Autorização - o que o usuário PODE fazer) """
    global DB_USERS
    try:
        with open('users.json', 'r', encoding='utf-8') as f: 
            DB_USERS = json.load(f)
        print(f"Sucesso: 'users.json' carregado. {len(DB_USERS)} usuários autorizados.")
    except Exception as e:
        print(f"Erro fatal: Não foi possível carregar users.json: {e}")
        DB_USERS = {}

def load_biometrics():
    """ Carrega o 'measures.csv' (Autenticação - quem o usuário É) """
    global DB_BIOMETRIC
    DB_BIOMETRIC = biometricRecognition.load_database('measures.csv')
    if DB_BIOMETRIC.empty:
        print("="*50)
        print("ATENÇÃO: O arquivo 'measures.csv' está vazio ou não foi encontrado.")
        print("Execute o script 'python dataBase.py' primeiro para cadastrar as mãos.")
        print("="*50)
    else:
        # Garante que a coluna 'person' (agora '01', '02') seja lida como string
        DB_BIOMETRIC['person'] = DB_BIOMETRIC['person'].astype(str)
        print(f"Sucesso: 'measures.csv' carregado. {len(DB_BIOMETRIC)} registros biométricos.")


load_permissions()
load_biometrics()


# --- Definição das Páginas (HTML embutido para simplicidade) ---
HTML_LOGIN = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <title>Sistema de Acesso</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }
        .container { max-width: 700px; margin: auto; padding: 20px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .erro { color: #D8000C; background-color: #FFD2D2; padding: 10px; border-radius: 5px; margin-bottom: 15px; }
        .sucesso { color: #2F6F2F; background-color: #DFF2BF; padding: 10px; border-radius: 5px; margin-bottom: 15px; }
        .grid-login { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 15px; margin-top: 20px; }
        
        .botao-com-imagem { 
            background-color: #f9f9f9; 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            padding: 10px; 
            text-align: center; 
            cursor: pointer; 
            transition: box-shadow 0.2s; 
            font-size: 14px;
            color: #333; /* Cor do texto do cargo */
            width: 100%;
            height: 150px; /* Altura fixa */
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .botao-com-imagem:hover { box-shadow: 0 0 10px rgba(0,0,0,0.15); }
        .botao-com-imagem img { 
            max-width: 100%; 
            height: 90px; 
            object-fit: cover; 
            display: block; 
            margin: 0 auto 10px; 
            border-radius: 4px; 
        }
        .botao-com-imagem span { 
            display: block; 
            font-weight: bold; 
            color: #333; /* Cor do Cargo */
        }
        
        /* NOVO: Estilo para o Header da Página de Login */
        .header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            border-bottom: 1px solid #eee; /* Linha de separação */
            padding-bottom: 10px;
        }
        .header h2 { margin: 0; }
        .header a.admin { 
            text-decoration: none; 
            background-color: #007bff; 
            color: white; 
            padding: 8px 12px; 
            border-radius: 5px; 
            font-size: 14px;
            flex-shrink: 0; /* Impede que o botão encolha */
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
             <h2>Sistema de Acesso - Ministério do Meio Ambiente</h2>
             <a href="/admin" class="admin">Painel Admin</a>
        </div>
    
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="{{ 'sucesso' if category == 'success' else 'erro' }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <h3 style="margin-top: 20px;">Simulador de Autenticação Biométrica</h3>
        <p>Selecione um usuário cadastrado para simular o login:</p>
        
        <div class="grid-login">
            {% for user in user_list %}
            <div>
                <form action="/authenticate" method="post" style="margin:0;">
                    <input type="hidden" name="user_id" value="{{ user.person_id }}">
                    <button type="submit" class="botao-com-imagem">
                        <img src="{{ url_for('serve_hand_db_image', subpath=user.url_subpath) }}" alt="{{ user.display_name }}">
                        <span>Usuário "{{ user.display_name }}"</span>
                    </button>
                </form>
            </div>
            {% endfor %}
            {% if not user_list %}
                <p>Nenhum usuário registrado. Execute 'python dataBase.py' para popular o sistema.</p>
            {% endif %}
        </div>
        
        </div>
</body>
</html>
"""

HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <title>Dashboard - Acesso Restrito</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }
        .container { max-width: 800px; margin: auto; padding: 20px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .header { display: flex; justify-content: space-between; align-items: center; }
        .header-nav a { text-decoration: none; background-color: #dc3545; color: white; padding: 8px 12px; border-radius: 5px; }
        .nivel { border: 1px solid #ccc; border-radius: 5px; padding: 15px; margin-top: 15px; }
        .nivel-1 { background-color: #e6f7ff; border-color: #b3e0ff; }
        .nivel-2 { background-color: #FFFFE0; border-color: #E6DB55; }
        .nivel-3 { background-color: #FFECEC; border-color: #D12F2F; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Dashboard - Informações Estratégicas</h2>
            <div class="header-nav">
                <a href="/logout">Sair</a>
            </div>
        </div>
        <p>Bem-vindo(a), <strong>{{ user.role }}</strong> (Nível de Acesso: {{ user.access_level }})</p>
        <hr>
        
        {% if user.access_level == 1 %}
        <div class="nivel nivel-1">
            <h3>Nível 1 (Acesso Público)</h3>
            <p>Informações públicas sobre as propriedades rurais que utilizam agrotóxicos.</p>
        </div>
        {% endif %}
        
        {% if user.access_level == 2 %}
        <div class="nivel nivel-2">
            <h3>Nível 2 (Acesso de Diretores)</h3>
            <p><strong>[RESTRITO]</strong> Relatórios sobre propriedades que utilizam agrotóxicos (nível divisão).</p>
        </div>
        {% endif %}

        {% if user.access_level == 3 %}
        <div class="nivel nivel-3">
            <h3>Nível 3 (Acesso do Ministro)</h3>
            <p><strong>[SECRETO]</strong> Informações estratégicas sobre propriedades que utilizam agrotóxicos proibidos e seus impactos nos lençóis freáticos, rios e mares.</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

HTML_ADMIN = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <title>Painel Admin</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }
        .container { max-width: 900px; margin: auto; padding: 20px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .header { display: flex; justify-content: space-between; align-items: center; }
        .header a { text-decoration: none; background-color: #007bff; color: white; padding: 8px 12px; border-radius: 5px; }
        .report-section { margin-top: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .file-list { list-style-type: none; padding: 0; }
        .file-list li { background-color: #f9f9f9; border: 1px solid #eee; padding: 5px; margin-top: 5px; }
        .admin-section { background-color: #f5f5f5; border: 1px dashed #ccc; padding: 15px; margin-top: 30px; border-radius: 5px; }
        input[type=submit] { background-color: #28a745; color: white; padding: 8px 12px; border: none; border-radius: 5px; cursor: pointer; }
        .erro { color: #D8000C; background-color: #FFD2D2; padding: 10px; border-radius: 5px; margin-bottom: 15px; }
        .sucesso { color: #2F6F2F; background-color: #DFF2BF; padding: 10px; border-radius: 5px; margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="container">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="{{ 'sucesso' if category == 'success' else 'erro' }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <div class="header">
            <h2>Painel de Administração</h2>
            <a href="/">Voltar ao Login</a> </div>
        <hr>

        <div class="admin-section">
            <h3>Adicionar Nova Mão</h3>
            <p>Envie uma nova imagem de mão (da pasta 'entryTest' ou do seu computador). O sistema irá verificar a legibilidade e, se aprovada, irá registrá-la na pasta 'working' e reconstruir o banco de dados. Imagens ilegíveis (mão esquerda, dedos colados, etc.) serão movidas para 'notWorking'.</p>
            <form action="/register" method="post" enctype="multipart/form-data">
                <input type="file" name="new_image" accept="image/*" required>
                <input type="submit" value="Verificar e Adicionar">
            </form>
        </div>

        <div class="report-section">
            <h3>Relatório: Usuários Autorizados (de users.json)</h3>
            <table>
                <thead>
                    <tr>
                        <th>ID do Usuário (person)</th>
                        <th>Cargo</th> 
                        <th>Nível de Acesso</th>
                    </tr>
                </thead>
                <tbody>
                    {% for user in authorized_users %}
                    <tr>
                        <td>{{ user.id }}</td>
                        <td>{{ user.role }}</td> 
                        <td>{{ user.level }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="report-section">
            <h3>Relatório: Imagens Rejeitadas (de handDatabase/notWorking)</h3>
            {% if rejected_images %}
                <ul class="file-list">
                    {% for image in rejected_images %}
                        <li>{{ image }}</li>
                    {% endfor %}
                </ul>
            {% else %}
                <p>Nenhuma imagem foi rejeitada. A pasta 'notWorking' está vazia.</p>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

# --- Rotas da Aplicação (Endpoints) ---

@app.route('/')
def index():
    """Página de Login"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    # --- LÓGICA DE CARREGAMENTO DOS BOTÕES (CORRIGIDA) ---
    # Mostra os usuários JÁ CADASTRADOS (da pasta 'working')
    
    if DB_BIOMETRIC.empty:
        unique_persons = []
    else:
        DB_BIOMETRIC['person'] = DB_BIOMETRIC['person'].astype(str)
        unique_persons = DB_BIOMETRIC['person'].unique()

    template_list = []
    
    for person_id in unique_persons: # person_id é "01", "02"...
        
        user_data = DB_USERS.get(person_id, {})
        role = user_data.get('role', f"ID {person_id} (Sem Cargo)")
        file_prefix = user_data.get('file_prefix') # Pega o "a", "b", etc.
        
        if file_prefix:
            # Procura a imagem na pasta 'working' (UPLOAD_FOLDER)
            found_image = glob.glob(os.path.join(UPLOAD_FOLDER, f"{file_prefix}_*.jpg")) + \
                                 glob.glob(os.path.join(UPLOAD_FOLDER, f"{file_prefix}_*.png")) + \
                                 glob.glob(os.path.join(UPLOAD_FOLDER, f"{file_prefix}_*.jpeg"))
            
            if found_image:
                f_path = found_image[0]
                f_name = os.path.basename(f_path)
                filename_no_ext = os.path.splitext(f_name)[0] # "a_001_001"
                
                # Cria o nome de exibição (ex: "a_001")
                display_name = filename_no_ext
                try:
                    parts = filename_no_ext.split('_')
                    if len(parts) >= 2:
                        display_name = f"{parts[0]}_{parts[1]}" # "a_001"
                except:
                    pass 

                try:
                    # Caminho relativo para a URL (ex: "working/a_001_001.jpg")
                    url_subpath = os.path.relpath(f_path, 'handDatabase').replace(os.path.sep, '/')
                except ValueError:
                    url_subpath = os.path.basename(f_path) # Fallback

                template_list.append({
                    "person_id": person_id,
                    "role": role, 
                    "display_name": display_name, 
                    "caminho_completo": f_path,
                    "url_subpath": url_subpath
                })
        else:
            print(f"Aviso: Usuário {person_id} está no CSV mas não no users.json ou não tem 'file_prefix'.")
    
    template_list.sort(key=lambda x: x['display_name']) 
    
    return render_template_string(HTML_LOGIN, 
                                  user_list=template_list)

@app.route('/handDatabase/<path:subpath>')
def serve_hand_db_image(subpath):
    """
    Serve com segurança os arquivos de imagem da pasta 'handDatabase'
    """
    return send_from_directory('handDatabase', subpath)

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """
    MUDANÇA: Esta é agora uma SIMULAÇÃO de login.
    NENHUM ARQUIVO É MOVIDO.
    """
    # MUDANÇA: Recebe o ID do usuário (ex: "01") diretamente.
    user_id = request.form.get('user_id')
    
    if not user_id:
        flash('ID de usuário inválido.', 'erro')
        return redirect(url_for('index'))

    # Caso 1: ID está autorizado no JSON
    if user_id in DB_USERS:
        print(f"Usuário '{user_id}' selecionado. Login OK.")
        session['user_id'] = user_id
        flash(f"Autenticação simulada bem-sucedida (Usuário {user_id})!", 'sucesso')
        return redirect(url_for('dashboard'))
    
    # Caso 2: ID não está autorizado (erro de sistema)
    else:
        print(f"Usuário '{user_id}' selecionado, mas NÃO ESTÁ AUTORIZADO (sem entrada no users.json).")
        flash(f"Mão reconhecida (Usuário {user_id}), mas este usuário não possui permissão de acesso.", 'erro')
        return redirect(url_for('index'))


@app.route('/register', methods=['POST'])
def register():
    """
    Lógica de "Triagem": Recebe uma nova imagem, testa a legibilidade
    (incluindo mão esquerda, dedos colados, etc.)
    e a move para 'working' (se legível) ou 'notWorking' (se ilegível).
    """
    
    # A verificação de admin (Nível 3) foi removida.
    
    if 'new_image' not in request.files:
        flash('Nenhum arquivo enviado.', 'erro')
        return redirect(url_for('admin_page'))
    
    file = request.files['new_image']
    
    if file.filename == '':
        flash('Nenhum arquivo selecionado.', 'erro')
        return redirect(url_for('admin_page'))

    if file:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(TEMP_FOLDER, filename)
        os.makedirs(TEMP_FOLDER, exist_ok=True)
        file.save(temp_path)
        print(f"Imagem temporária salva em: {temp_path} para verificação...")

        try:
            print("Verificando legibilidade da imagem...")
            # Esta função agora retorna None para mãos ilegíveis (dedos colados, etc.)
            features, _, _ = biometricRecognition.processImage(temp_path, output_dir=None)

            if features is None:
                print(f"...Imagem '{filename}' não é legível. Movendo para '{NOT_WORKING_FOLDER}'.")
                os.makedirs(NOT_WORKING_FOLDER, exist_ok=True) 
                not_working_path = os.path.join(NOT_WORKING_FOLDER, filename)
                os.rename(temp_path, not_working_path) 
                
                # MUDANÇA: Mensagem de erro genérica
                flash(f"Erro: Imagem '{filename}' não é legível (ex: mão esquerda, dedos colados, iluminação ruim). Imagem movida para '{NOT_WORKING_FOLDER}'.", 'erro')
                return redirect(url_for('admin_page'))

            print(f"...Imagem '{filename}' legível. Movendo para '{UPLOAD_FOLDER}'.")
            permanent_path = os.path.join(UPLOAD_FOLDER, filename)
            os.rename(temp_path, permanent_path)
            
            # 4. Dispara a RECONSTRUÇÃO TOTAL (que atualiza CSV e JSON)
            print("Disparando reconstrução do banco de dados (dataBase.main())...")
            dataBase.main() 
            
            # 5. Recarrega AMBOS os bancos de dados na aplicação
            print("Recarregando 'measures.csv' na aplicação...")
            load_biometrics() 
            print("Recarregando 'users.json' na aplicação...")
            load_permissions()
            
            flash(f"Imagem '{filename}' é legível e foi adicionada com sucesso! O banco de dados foi re-processado.", 'sucesso')
        
        except Exception as e:
            print(f"Erro ao registrar nova imagem: {e}")
            flash(f"Erro ao registrar imagem: {e}", 'erro')
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return redirect(url_for('admin_page'))


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Você precisa se autenticar primeiro.', 'erro')
        return redirect(url_for('index'))
    
    user_id = session['user_id']
    if user_id not in DB_USERS:
        session.clear() 
        flash('Erro: Seu usuário foi removido do sistema.', 'erro')
        return redirect(url_for('index'))

    user_data = DB_USERS[user_id]
    
    return render_template_string(HTML_DASHBOARD, user=user_data)

@app.route('/admin')
def admin_page():
    """
    Página de Admin (Relatórios) - Atualizada para remover a coluna 'prefixo'
    """
    authorized_users = []
    load_permissions() 
    
    for user_id, data in DB_USERS.items():
        authorized_users.append({
            "id": user_id, 
            "role": data.get('role', 'N/A'),
            "level": data.get('access_level', 'N/A')
            # A chave "prefixo" não é mais exibida
        })
    
    # Relatório 2: Escaneia a pasta 'notWorking'
    rejected_images = []
    extensions = ('*.jpg', '*.png', '*.jpeg')
    os.makedirs(NOT_WORKING_FOLDER, exist_ok=True)
    for ext in extensions:
        files = glob.glob(os.path.join(NOT_WORKING_FOLDER, ext))
        for f in files:
            rejected_images.append(os.path.basename(f))

    return render_template_string(HTML_ADMIN, 
                                  authorized_users=authorized_users, 
                                  rejected_images=rejected_images)

@app.route('/logout')
def logout():
    session.clear() 
    flash('Você saiu do sistema.', 'sucesso')
    return redirect(url_for('index'))

# --- Execução da Aplicação ---
if __name__ == '__main__':
    # Garante que todas as pastas necessárias existem
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(NOT_WORKING_FOLDER, exist_ok=True) 
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    os.makedirs(ENTRY_TEST_FOLDER, exist_ok=True)

    print("="*50)
    print("Aplicação APS - Sistema de Autenticação Biométrica")
    print(f"Imagens de login (botões) lidas de: {UPLOAD_FOLDER}")
    print(f"Imagens de DB (cadastradas) estão em: {UPLOAD_FOLDER}")
    print(f"Imagens de teste (para upload) estão em: {ENTRY_TEST_FOLDER}")
    print(f"Imagens rejeitadas (ilegíveis) serão salvas em: {NOT_WORKING_FOLDER}")
    
    # Imprime os status de carregamento iniciais
    load_permissions()
    load_biometrics()

    print(f"Threshold de Distância: {AUTH_THRESHOLD}")
    print("Servidor iniciando em http://127.0.0.1:5000")
    print("="*50)
    app.run(debug=True, port=5000, host='0.0.0.0')