# Guia de contribuição (para o grupo)

## Fluxo de trabalho

1. **Nunca commite direto na `main`.** Crie uma branch por tarefa:
   ```bash
   git checkout main && git pull
   git checkout -b feat/nome-da-tarefa     # ex.: feat/smote, fix/readme, docs/resultados
   ```
2. Faça commits pequenos e com mensagem clara, no imperativo:
   `Adiciona SMOTE ao pipeline`, `Corrige escala de Amount`.
3. Envie a branch e abra um **Pull Request**:
   ```bash
   git push -u origin feat/nome-da-tarefa
   ```
4. Peça revisão a pelo menos **1 colega** antes de fazer o merge.
5. Antes de abrir o PR, atualize sua branch: `git pull origin main`.

## Dicas

- **Pegue uma issue** (aba *Issues*) e se atribua antes de começar, para ninguém trabalhar na mesma coisa.
- **Notebooks:** evite dois membros editando o mesmo `.ipynb` ao mesmo tempo (conflitos são chatos). Prefira um notebook por pessoa, ex.: `notebooks/02_nome_experimentos.ipynb`. Limpe as saídas pesadas antes do commit.
- **Não suba** o dataset nem o `model.joblib` (já estão no `.gitignore`).
- **Novo modelo?** Adicione em `src/models.py`, na função `get_models`, e ele entra automaticamente na comparação.
- Mantenha as sementes aleatórias (`seed=42`) para os resultados serem reproduzíveis.
