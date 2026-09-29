# Python Virtual Environment Setup - Quick Notes

## Create Virtual Environment
python -m venv venv

## Activate
# Windows CMD
venv\Scripts\activate

# Windows PowerShell
venv\Scripts\Activate.ps1

## Install Dependencies
pip install langchain langchain-openai openai

## Save Dependencies
pip freeze > requirements.txt

## Install from requirements.txt
pip install -r requirements.txt

## Run the Script
python langchain_graph\langchain_example.py

## Deactivate
deactivate

---

## Quick Start (copy-paste)
python -m venv venv
venv\Scripts\activate
pip install langchain langchain-openai openai
python langchain_graph\langchain_example.py

## Troubleshooting
# If PowerShell blocks activation:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
