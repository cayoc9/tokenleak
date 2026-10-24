# Por que manter acurácia e acrescentar outras métricas

A pergunta científica decide a métrica. Top-1 por token mede classificação de posições; não mede diretamente recuperação de prompts inteiros ou exposição de informação protegida. Sua fórmula é posições corretas divididas por todas as posições avaliadas. Top-5 mede inclusão do token verdadeiro numa lista de cinco candidatos por posição. Nenhuma delas é a porcentagem de prompts integralmente reconstruídos.

## Dois contraexemplos ilustrativos

Uma sequência de dez posições contém nove tokens comuns e um token protegido. Recuperar os nove primeiros e errar o último dá 90% de acurácia e zero recuperação integral do trecho protegido. Outra sequência tem um trecho protegido de três tokens: recuperar apenas esse trecho dá 30% de acurácia e 100% de recuperação integral do segredo dessa sequência. Esses exemplos são testes construídos, não observações de um ataque real. Tokens BPE também não equivalem necessariamente a palavras.

## Escolha de métricas por afirmação

| Afirmação | Métrica e denominador | Motivo e limite |
|---|---|---|
| Classifico posições de tokens | Top-1 e Top-5 sobre todas as posições | Conserva a ligação com o artigo; Top-5 ainda contém ambiguidade e não é uma sequência única recuperada. |
| Reconstruo prompts inteiros | Prompts com todas as posições Top-1 corretas / prompts de teste | Evidência direta de recuperação exata; rigorosa e dependente do comprimento. Reportar por faixa de tamanho. |
| Reconstruo parcialmente a sequência | Distância de Levenshtein / comprimento de referência, média por prompt | Conta inserções, remoções e substituições; especificar se a unidade é token ou caractere. Neste pipeline a unidade é token, com uma previsão por posição conhecida. |
| Exponho um campo protegido | Trechos protegidos integralmente recuperados / trechos anotados | Aproxima o objetivo de privacidade quando a anotação identifica realmente o alvo. Aqui são apenas palavras sintéticas marcadas, não PII. |
| Tenho bom desempenho além dos tipos frequentes | Recall médio por tipo e acerto nos tipos raros | Um tipo raro passa a ter peso semelhante ao comum na média por tipo; poucos exemplos tornam a estimativa instável. Sempre informar suporte. |
| O desempenho vale para um prompt típico | Média dos acertos de cada prompt | Evita que prompts longos dominem a média; não substitui a média por tipo. |
| Identifico tipos sem excesso de falsos positivos | Macro-F1 por tipo | Combina precisão/recall por classe, mas não preserva ordem nem identifica relevância privada. Micro-F1 não acrescenta informação à acurácia neste caso multiclasse de um rótulo por posição. |

Exatidão de sequência e de segredo também precisam de limites: errar apenas uma pontuação pode zerar o prompt exato, enquanto um vazamento parcial ainda pode causar dano; um segredo composto pode ser sensível mesmo sem recuperação integral. Nosso proxy atual de palavras protegidas é uma primeira infraestrutura, não uma definição definitiva de dano. Uma próxima versão deve acrescentar campos sintéticos realistas, estratificação por comprimento/tipo de campo e medidas de recuperação parcial, sem dados pessoais reais.

## O que F1 e similaridade semântica não resolvem

Usar macro-F1 ajuda no desequilíbrio, mas não transforma classificação em avaliação de privacidade. F1 ponderado pelo suporte pode continuar dominado pelas classes frequentes. Similaridade semântica é auxiliar: uma frase com outro nome, outra senha ou outro valor pode continuar muito semelhante e estar errada justamente no alvo relevante. Scores TRCS também não devem ser interpretados como probabilidades; medir calibração demanda uma etapa própria.

## Como interpretar nossos resultados

As condições compartilham os mesmos prompts e sinais de teste. Comparar fusão com timing/cache isolados e com controles de frequência, sinais constantes e sinais permutados. Se o controle preserva quase o mesmo desempenho do ataque, investigar informação no prior, desenho do corpus ou vazamento no avaliador antes de atribuir ganho ao canal. Um controle permutado pode acertar por acaso; não se espera obrigatoriamente zero.

ICs atuais se referem a dez sementes com corpus e canal latente fixos. A unidade é a rodada de medição, não cada token independente. Bons percentuais em assinaturas sintéticas por token mostram separabilidade desse canal construído; não comprovam que hardware real emita essas assinaturas. Custos de perfilamento e as hipóteses de posição conhecida/vocabulário fechado precisam acompanhar os percentuais.

## Fontes primárias

- [Documentação de avaliação do scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html): relação entre objetivo, acurácia, balanced accuracy e médias de F1; consulta em 23/10/2026.
- [Gao et al., USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/gao-zibo): exemplo do campo que reporta distância de edição e similaridade para reconstrução; não torna o seu protocolo equivalente ao TokenLeak.

As recomendações de privacidade e a hierarquia de métricas acima são decisões metodológicas propostas para este projeto, não um consenso de métrica universal nem uma alteração do texto canônico do artigo.
