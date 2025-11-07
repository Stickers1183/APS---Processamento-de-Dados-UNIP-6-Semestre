# 🖐️ Projeto APS - Reconhecimento Biométrico por Geometria da Mão

Sistema de Controle de Acesso **simulado** desenvolvido para a disciplina de **Processamento de Imagem e Visão Computacional (PIVC)**.

---

## 🎯 Tema do Projeto

O projeto simula um **sistema de segurança de alto nível** para o **Ministério do Meio Ambiente**.  
O sistema utiliza a **geometria da mão** como forma de autenticação biométrica para liberar (ou negar) o acesso a diferentes níveis de informação sigilosa.

---

## ✨ Funcionalidades

- **Geração de Banco de Dados:**  
  Um script (`dataBase.py`) processa imagens de mãos e gera um banco de dados biométrico (`measures.csv`) e um de permissões (`users.json`).

- **Interface Web:**  
  Um painel em Flask (`app.py`) que simula a tela de login do sistema.

- **Simulação de Autenticação:**  
  O login é simulado (clicando no botão do usuário) para demonstrar o controle de acesso.

- **Níveis de Acesso:**  
  O sistema demonstra 3 níveis de acesso:
  - Nível 1: Público  
  - Nível 2: Diretor  
  - Nível 3: Ministro

- **Painel de Admin:**  
  Uma rota `/admin` permite o envio de novas imagens de mãos.

- **Verificação de Imagem (Triagem):**  
  O sistema valida se a nova imagem é “legível” (mão direita, dedos abertos, etc.) antes de aceitá-la e reprocessar o banco de dados.

---

## 💻 Tecnologias Utilizadas

- **Python 3**
- **Flask** — Servidor web e interface do sistema.
- **OpenCV (cv2)** — Processamento de imagem, contornos e medições.
- **Pandas** — Manipulação do banco de dados biométrico (`measures.csv`).
- **NumPy** — Cálculos matemáticos e de vetores.
- **Imageio** — Leitura dos arquivos de imagem.

---

## 📂 Estrutura de Arquivos

```
/projeto-aps-biometria
│
├── app.py                  # Script principal (Servidor Flask)
├── dataBase.py             # Script para (re)gerar o banco de dados
├── biometricRecognition.py  # "Motor" de Visão Computacional (PIVC)
├── requirements.txt         # Lista de dependências (pip install -r)
├── .gitignore               # Arquivos a serem ignorados pelo Git
│
├── handDatabase/
│   ├── entryTest/          # (Vazia) Imagens de exemplo para testar o upload
│   ├── working/            # (Pré-populada) Imagens VÁLIDAS para o DB
│   ├── notWorking/         # (Vazia) Onde imagens ruins são movidas
│   ├── temp_uploads/       # (Vazia) Pasta temporária para uploads
│
├── measures.csv            # (Gerado) Banco de dados biométrico
└── users.json              # (Gerado) Banco de dados de permissões e cargos
```

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos

- [Python 3.7+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)

---

### 2. Clonar o Repositório

```bash
# Substitua pela URL do seu repositório
git clone https://github.com/Stickers1183/APS-Processamento-de-Dados-UNIP-6-Semestre.git.git
cd [APS-Processamento-de-Dados-UNIP-6-Semestre]
```

---

### 3. Criar Ambiente Virtual e Instalar Dependências

É altamente recomendado usar um ambiente virtual (`venv`) para isolar as dependências.

```bash
# Opcional - Criar o ambiente virtual
python -m venv venv

# Ativar o ambiente virtual
# No Windows (PowerShell/CMD):
.env\Scriptsctivate
# No macOS/Linux:
source venv/bin/activate

# Instalar todas as dependências
pip install -r requirements.txt
```

---

### 4. Popule a Pasta `working`

O projeto precisa de imagens iniciais para construir o primeiro banco de dados.

Certifique-se de que a pasta `handDatabase/working/` contém as imagens de mão válidas  
(ex: `a_001_001.jpg`, `b_002_001.jpg`, `c_003_001.jpg`, etc.).

---

### 5. Gere o Banco de Dados

Antes de rodar a aplicação web, gere os arquivos `measures.csv` e `users.json`:

```bash
python dataBase.py
```

---

### 6. Execute a Aplicação Web

Agora que o banco de dados existe, inicie o servidor Flask:

```bash
python app.py
```

O terminal deverá mostrar:

```
 * Servidor iniciando em http://127.0.0.1:5000
```

---

### 7. Acesse o Sistema

Abra seu navegador e acesse:  
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📋 Modo de Uso (Passo a Passo para o Professor)

### 1. Teste de Login (Simulado)

1. Acesse `http://127.0.0.1:5000`.  
2. Clique no botão do “Ministro” (ex: Usuário `a_001`).  
3. Você será autenticado e redirecionado para o **Dashboard**.  
4. Verifique que o acesso é **Nível 3 (Secreto)**.  
5. Faça logout e entre como “Público” (ex: Usuário `c_003`).  
6. Este usuário terá acesso apenas ao **Nível 1 (Público)**.

---

### 2. Teste do Painel de Admin e Triagem de Imagens

Acesse o painel de administração:  
👉 [http://127.0.0.1:5000/admin](http://127.0.0.1:5000/admin)

Você verá os relatórios de “Usuários Autorizados” e “Imagens Rejeitadas”.

---

#### 🧪 Teste 2.1: Adicionar uma Mão Inválida (Ex: Dedos Dobrados)

1. Em “Adicionar Nova Mão”, clique em **Escolher arquivo**.  
2. Selecione uma imagem inválida (ex: `a_001_005.jpg` da pasta `entryTest/`).  
3. Clique em **Verificar e Adicionar**.  

O sistema deve exibir:

```
Erro: Imagem 'a_001_005.jpg' não é legível...
```

A imagem será movida para `handDatabase/notWorking/`  
e aparecerá no relatório de **Imagens Rejeitadas**.

---

#### ✅ Teste 2.2: Adicionar uma Mão Válida (Novo Usuário)

1. Envie uma imagem válida (ex: `d_001_001.jpg` da pasta `entryTest/`).  
2. Clique em **Verificar e Adicionar**.  

O sistema mostrará:

```
Imagem 'd_001_001.jpg' é legível e foi adicionada com sucesso! 
O banco de dados foi re-processado.
```

O script `dataBase.py` será executado automaticamente.  
O relatório de **Usuários Autorizados** exibirá o novo usuário (`Usuário 04`, cargo “Público”).  

Volte à tela de login e verifique que o novo botão `Usuário 'd_001'` foi adicionado.

---

## 👥 Autores

- Lucas Pedro Américo da Silva RA: T881HA3
- Kaiky Souza Proença de Andrade RA: N088575
- Gustavo Dias de Oliveira RA: G7862G6

---

📘 **Disciplina:** Processamento de Imagem e Visão Computacional (PIVC)  
🏫 **Instituição:** UNIP - UNIVERSIDADE PAULISTA
📅 **Ano:** 2025