# GeoRede Ajuste

Transformação de coordenadas XYZ por estação e ajustamento conjunto de rede.
O perfil executa SVD-P, SCH-Q, Q-ST, CMTLS, TFS e N-LM.

## Executar

Use Python 3.12 em um ambiente virtual. Instale as dependências e execute:

```text
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run.py
```

Os arquivos de entrada estão em `data/`. As saídas numéricas são criadas em
`output/run_01/`. 

Com `.venv` configurado, `python run.py` também seleciona automaticamente esse
ambiente, mesmo que o Python padrão do Windows seja outra versão. Não publique
a pasta `.venv`: ela é específica deste computador e está no `.gitignore`.

No Windows, use `executar.cmd` para abrir diretamente o ambiente local,
evitando mensagens de pacotes quebrados em outras instalações de Python.
O arquivo chama o mesmo `run.py`, sem alterar o processamento. A janela
permanece aberta ao final para leitura do resultado.

```text
python run.py --out output/run_02
```

Para outra pasta de entrada:

```text
python run.py --data caminho/dos/dados --out output/novo_levantamento
```

Preserve o formato dos sete arquivos de exemplo, incluindo a identificação de
controles e ICPs. A verificação realizada cobre esse levantamento, não valida
automaticamente novos dados ou geometrias.

## Organização

```text
run.py             comando único de execução
requirements.txt   dependências numéricas
data/              arquivos XYZ de entrada
config/            parâmetros de base
src/               processamento e módulos numéricos internos
```

Execute apenas `run.py`. Os arquivos de `src/` são dependências internas;
não representam fluxos alternativos para reproduzir o artigo. Testes, auditorias,
gráficos, scripts editoriais e resultados arquivados ficam fora desta distribuição.

## Perfil metodológico

N-LM com amortecimento adaptativo, até 50 iterações, tolerância de incremento
1e-8 e gradiente relativo 1e-7. Métodos por estação: até 25 iterações quando
iterativos. Baarda somente na rede, com |w| > 3 e desvio padrão a priori
**assumido** de 2 mm; MAD > 3,5, mínimo de oito normas para exclusão somente
por MAD. O mínimo operacional de suporte é quatro controles compartilhados
por estação. A triagem externa permite até 20 remoções e verifica a solução
proposta antes de confirmar a exclusão na rede. A remoção interna do kernel
fica desativada. Pesos unitários; MDB com potência teórica de 80%; MIB não calculado.

As opções explícitas de `run.py` prevalecem sobre `config/base.json`.
O manifesto de cada execução registra a configuração efetiva. Os limiares
internos de amortecimento e gradiente estão implementados no núcleo N-LM.

CP43 apresenta quase empate entre TS03 e TS06: pequenas diferenças numéricas
podem alternar a ocorrência selecionada e o número de iterações. Isso não
identifica qual estação contém erro grosseiro. A regra original foi preservada.
ICPs são excluídos da estimação, mas sua referência vem do mesmo levantamento;
não constituem verdade externa. Q-ST é a variante implementada com matriz livre
afim. Campos complementares exportados não são critérios adicionais de remoção.

