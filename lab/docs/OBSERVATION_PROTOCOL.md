# Protocolo M5: observação real de inferência em laboratório

Este piloto mede um serviço GPT-2 pequeno em CPU e investiga se os tempos da resposta ajudam a distinguir **um token de entrada entre oito candidatos públicos**, em uma posição pública e fixa. É uma condição própria de viabilidade, distinta do GPT-2 XL do artigo, da simulação SPC-5K e da reconstrução integral de prompts.

## Segredo, observador e informação disponível

O coordenador prepara um prompt controlado de dezesseis tokens, com o candidato secreto na posição oito. Quatro contextos públicos e oito palavras candidatas têm tokenização fixa. O coordenador conhece a verdade para o experimento; o classificador recebe somente seis características de timing e transporte extraídas de `observations.jsonl`. IDs são UUID4 opacos. Rótulos, texto, IDs de entrada, contexto, ordem, sessão e diagnóstico interno não são características do classificador.

A visão de observação é instrumental: o coordenador envia a entrada privada e mede a resposta, depois fornece ao classificador uma visão sem a entrada. Não foi implementado um adversário separado capturando passivamente o fluxo de uma vítima. A disponibilidade de uma visão equivalente ao adversário de uma implantação precisa ser demonstrada em outro protocolo; reter rótulos do classificador não estabelece esse acesso.

O serviço é acessível apenas em `127.0.0.1`, dentro de um contêiner sem rede externa. O observador mede recebimentos reais do socket. Requisições têm 2.048 bytes, com padding; cada passo de saída emite um marcador fixo de 64 bytes, sem o valor do token. TCP_NODELAY é declarado nos dois lados. Isso isola um canal de tempo em um serviço de laboratório; não corresponde automaticamente a um endpoint de produção, metadados TLS ou tráfego WAN.

Um marcador é emitido por passo de geração, mas uma leitura do socket pode conter vários marcadores. O coletor preserva essa coalescência e atribui aos marcadores completados na mesma leitura o mesmo instante observado. Não usa o número de leituras como número de tokens. O alvo é inferido da resposta agregada; não se associa um tempo de token de saída a uma posição secreta de entrada.

## Modelo e orçamento registrados

- Modelo: `openai-community/gpt2`, revisão `607a30d783dfa663caf39e06633721c8d4cfcd7e`, safetensors verificado por SHA-256, CPU/float32.
- PyTorch 2.3.1+cpu, Transformers 4.41.2, Python 3.11.9; parâmetros e freeze completos acompanham cada execução.
- Dezesseis tokens de entrada, oito passos de geração gulosa e KV cache habilitado. Não há coleta física de cache.
- Três sessões com novos processos/modelos, quatro contextos, oito classes e oito repetições balanceadas: 768 solicitações por condição.
- Três condições: GPT-2 com a entrada experimental; GPT-2 com o candidato substituído por um token constante; servidor sem inferência. A ordem das condições varia por sessão com seed registrada.
- Vinte solicitações de aquecimento por processo, identificadas e excluídas do denominador de avaliação. Solicitações de avaliação não são descartadas silenciosamente.

## Fontes de timing

O observador registra chegada do primeiro e último marcador desde o início do envio da requisição, média/desvio dos intervalos, número de leituras com corpo e máximo de marcadores na mesma leitura. A conexão TCP é aberta antes desse relógio; sua abertura não integra a medida. O transporte, parsing do servidor, inferência e envio da resposta integram o intervalo.

Separadamente, o servidor registra duração de cada chamada do modelo com argmax: primeira chamada de prefill e chamadas subsequentes de decode. Esses dados são diagnóstico privilegiado, não características entregues ao adversário. Os relógios relativos do cliente/servidor não são subtraídos entre si. Em CPU, as chamadas medidas são síncronas; esta implementação não afirma aplicar o mesmo coletor a kernels CUDA assíncronos.

## Ajuste, partições e controles

Uma regressão logística multinomial com StandardScaler, C=1, pesos de classes balanceados e parâmetros fixos é ajustada somente na sessão zero/contextos X00/X01. A sessão um/contextos conhecidos é validação diagnóstica. O teste principal usa sessão dois e contextos X02/X03, separando sessão e contexto do ajuste.

Também são relatados novo contexto na mesma sessão e nova sessão em contextos conhecidos. A frequência do treino e o acaso uniforme são referências. O controle com entrada cegada testa se o sinal depende do candidato; o servidor sem inferência ajuda a investigar transporte e processamento da requisição. Desempenho elevado nesses controles exige investigar confundimento antes de atribuir a classificação ao modelo.

Top-1 é a métrica principal de classificação do candidato. Top-5, recall balanceado, macro-F1 e log loss são diagnósticos. Com oito classes, acaso é 12,5% Top-1 e 62,5% Top-5, diferente de um vocabulário com 8.431 tipos. Não se compara diretamente esse número com os 73,4% do artigo.

ICs usam bootstrap por blocos de repetição; o teste de randomização permuta rótulos dentro de contexto × repetição, mantendo o classificador fixo. São inferências condicionais ao pequeno desenho de laboratório, com oito blocos; não demonstram generalização entre máquinas. Nenhum hiperparâmetro é escolhido pelo teste.

## Evidência para avançar

Conferir cobertura completa, entradas de comprimento constante, segredo na posição declarada, observações sem rótulos, características recalculáveis, predições refeitas dos coeficientes e desempenho dos controles. Um resultado sem vantagem neste piloto limita a evidência a estas condições; não prova ausência de todos os canais possíveis. Um resultado positivo precisa resistir aos controles, a novas sessões e a repetições independentes antes de expandir o conjunto de candidatos ou alegar vazamento generalizável.

Fontes técnicas: [GPT-2/Transformers 4.41.2](https://huggingface.co/docs/transformers/v4.41.2/en/model_doc/gpt2), [checkpoint público GPT-2](https://huggingface.co/openai-community/gpt2), [binários CPU PyTorch](https://pytorch.org/get-started/previous-versions/). As escolhas de desenho acima são desta implementação, não propriedades recuperadas de uma execução histórica.
