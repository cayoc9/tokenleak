# Resultados da simulação independente v1

São dez sementes de medição, com corpus/canal latente fixos; sinais sintéticos e posições de entrada conhecidas. Não houve inferência de GPT-2 nesta condição. Os percentuais históricos de 73,4%/91,2% não foram reproduzidos.

| Condição | Top-1 token | Top-5 token | Prompt integral | Trecho sintético integral |
|---|---:|---:|---:|---:|
| Frequência do treino | 8.618% | 20.020% | 0.000% | 0.200% |
| Acaso uniforme | 0.008% | 0.052% | 0.000% | 0.000% |
| Timing | 0.037% | 0.208% | 0.000% | 0.000% |
| Cache | 99.475% | 99.876% | 83.170% | 98.270% |
| Fusão fixa | 55.996% | 60.811% | 0.020% | 18.370% |
| Fusão aprendida | 99.577% | 99.909% | 86.310% | 98.530% |
| Fusão com poda | 99.187% | 99.500% | 75.690% | 97.420% |
| Sinais constantes | 0.006% | 0.042% | 0.000% | 0.000% |
| Pares permutados | 1.250% | 1.302% | 0.000% | 0.000% |

O corpus novo tem 5.000 prompts controlados de palavras públicas, vinte categorias, 8–64 tokens, 178.209 posições totais e 8.431 tipos. Há 5.227 tipos com pelo menos dez ocorrências e 3.204 raros. Categorias rotulam instruções; palavras são compartilhadas entre elas. Não é um benchmark de conversação natural.

O split por famílias de instrução contém 3.000 treino, 1.000 validação e 1.000 teste. Cada rodada avalia 35.380 posições. O atacante conhece o vocabulário fechado e recebe 50 amostras de calibração por tipo, além dos textos de perfilamento: 421.550 amostras por rodada. Todos os tipos têm perfil; os tipos raros não sofrem escassez de calibração.

A fusão aprendida obteve 99,577% ± 0,024 ponto percentual Top-1 (IC95 entre sementes). O ganho pareado sobre cache isolado foi 0,102 ± 0,023 ponto. A poda perdeu 0,390 ± 0,029 ponto; conservou cerca de 5.906 candidatos em média, com recall de 99,579% e um fallback em 353.800 posições.

## Sensibilidade exploratória ao canal assumido

Nas mesmas sementes 0/1 e com corpus/split idênticos, limitar o gerador a 128 assinaturas de cache compartilhadas reduziu Top-1 da fusão de 99,577% para 4,638%; Top-5 foi 22,035% e nenhum prompt foi recuperado integralmente. Essa condição foi definida após observar a discriminação elevada da principal. Não é busca por parâmetros do artigo nem calibração de um cache físico.

O resultado alto depende do canal abstrato. Na condição de colisões, o classificador fica abaixo do prior de frequência de 8,618%. A avaliação evidencia uma limitação do decodificador sem prior explícito e a necessidade de medir acesso/colisões reais.

## Verificação e interpretação

O piloto de 200 prompts/512 tipos foi repetido com dezenove arquivos científicos idênticos. Retokenização integral, perfis refeitos das amostras, replay dos sinais, 810 conferências de métricas, trinta posições de ranking e noventa distâncias de edição amostradas passaram. O auditor compartilha o gerador para o replay; seus cálculos de métricas e ranking não importam o avaliador/reconstrutor.

Trechos protegidos são os últimos até três tokens de palavras, como proxies sintéticos. Sua recuperação não mede o valor de segurança de campos reais. ICs são condicionais ao corpus/canal e não estimam novos dispositivos ou sessões.

Os resultados completos estão em `lab/results/`; o corpus está em `lab/data/corpus-v1/`. Intermediários completos são distribuídos como artefato de release, com manifesto. Os fontes científicos de `src/`, dados e artigo recebidos do upstream foram preservados.
