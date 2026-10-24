# Piloto M5: tempos de resposta de GPT-2 pequeno em CPU

## Resultado principal

O piloto coletou 2.304 observações, com 768 requisições por condição, três sessões, quatro contextos, oito candidatos e oito repetições. O teste principal contém 128 requisições em sessão e contextos separados do treino. O aquecimento foi excluído por finalidade registrada, e nenhuma requisição de avaliação foi descartada.

| Condição | Top-1 | IC95 por blocos | Top-5 | p de randomização unilateral |
|---|---:|---:|---:|---:|
| GPT-2 com entrada experimental | 10.156% | 7.031–14.844% | 64.062% | 0.8776 |
| GPT-2 com entrada cegada | 9.375% | 5.469–14.062% | 66.406% | 0.9070 |
| Servidor sem inferência | 13.281% | 7.031–21.094% | 63.281% | 0.4193 |
| Acaso/prior balanceado | 12,500% | referência condicional | 62,500% | — |

**Não foi demonstrada vantagem de classificação no teste principal.** GPT-2 acertou 13 de 128 candidatos (10,156%), abaixo dos 12,5% do prior/acaso; seu p de randomização foi 0,8776. Os controles ficam na mesma faixa. Isso limita a evidência nestas condições; não prova ausência de todos os canais de timing, nem justifica aumentar o escopo ou ajustar parâmetros até obter resultado positivo.

## O que foi medido

O observador mede a chegada de marcadores de resposta de tamanho fixo, em TCP loopback, desde o início do envio da requisição e excluindo a abertura da conexão. Seus seis atributos são primeiro/último marcador, média/desvio dos intervalos, leituras com corpo e máximo de marcadores por leitura. Requisições são preenchidas a 2.048 bytes; respostas têm oito marcadores de 64 bytes. Coalescência de marcadores é registrada. O coletor não usa rótulos, valores dos tokens, contexto, IDs, ordem, sessão ou durações internas como atributos.

O alvo é um único token de entrada em posição pública oito, entre oito palavras públicas. O modelo é GPT-2 pequeno, CPU/float32, revisão `607a30d783dfa663caf39e06633721c8d4cfcd7e`, com pesos safetensors verificados, geração gulosa e KV cache apenas dentro da requisição. Essa condição é distinta de GPT-2 XL, de SPC-5K e de reconstrução integral do prompt. Não há probe físico de cache, rede externa, TLS ou serviço de terceiros.

O coordenador envia as entradas e instrumenta a resposta; o classificador recebe uma visão sem rótulos/conteúdo de entrada. Isso mede a classificabilidade dessa visão controlada. Não demonstra que um adversário passivo separado consiga obter a mesma observação de uma vítima em produção. A ausência de vantagem no piloto e a disponibilidade do canal ao adversário são questões distintas.

## Separação e controles

A regressão logística padronizada, C=1, foi ajustada somente na sessão zero/contextos X00/X01 (128 requisições por condição). A sessão um/contextos conhecidos é diagnóstica; a principal usa sessão dois/contextos X02/X03. Não houve escolha de parâmetros pelo teste. As duas partições adicionais de transferência e todas as métricas constam de `summary.json`.

A condição cegada substitui o candidato por um token constante antes da inferência, mantendo contexto e requisição de tamanho fixo; a condição sem inferência ajuda a detectar confundimento de transporte/parsing. A ordem das condições é sorteada por sessão; blocos de contexto/candidato são balanceados e aleatorizados a cada repetição.

ICs são bootstrap de oito blocos de repetição; o teste de randomização usa 2.000 permutações dentro de contexto × repetição com classificador fixo. A amostra é pequena e os intervalos são condicionais a candidatos, contextos, sessões e máquina. Não são intervalos para outros dispositivos ou usuários.

## Verificação

O smoke test completou 288 observações e 192 predições conferidas. A execução principal conferiu 2.304 observações e 1.536 predições nas quatro partições diagnósticas/principal; a auditoria externa recalculou características sem importar o extrator original, refez os três ajustes somente com o treino, verificou tamanhos fixos, posição secreta, balanceamento, linhagem de leituras/marcadores e contagens. Diagnósticos de servidor permanecem separados e não entram no classificador.

O código científico de coleta e avaliação nesta publicação coincide com o snapshot da execução. O wrapper recebeu uma checagem adicional de identidade/hash de assets após o primeiro piloto; essa barreira não muda os sinais coletados ou o ajuste. Os comandos, snapshots e manifestos permitem inspecionar a versão efetiva.

## Consequência metodológica

O número alto da simulação não se transportou para este observador real. Antes de qualquer alegação de reconstrução física, é necessário investigar informação dependente do segredo com controles, reproduzir o efeito em novas sessões/contextos e demonstrar o acesso efetivo do adversário. O próximo avanço deve ser guiado pelo resultado medido e por uma hipótese de canal, mantendo a referência negativa e os mesmos critérios de separação.

Fonte técnica do modelo: https://huggingface.co/openai-community/gpt2 . Desenho completo: `OBSERVATION_PROTOCOL.md`.
