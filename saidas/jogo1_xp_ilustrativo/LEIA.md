# eXP ilustrativo (NÃO é avaliação)

Mesmo pipeline de `../jogo1`, mas aceitando quadros com erro de reprojeção das linhas ≤ **15 px** (em vez de ≤ 3 px):
`filtrar_projecao.py --erro-max 15`. Com o critério rigoroso sobra 1 passe em metros e o `xpass.py` não roda.

Aqui sobram 14 passes — o suficiente para o script gerar `resultados_xp.csv`, `calibracao_xp.png` e as superfícies,
mas **pouco demais** para comparar modelos (ver README, Q6.2): 15 px podem significar metros de erro longe da câmera.
Entradas (tracks.csv, bola.csv, camera.npy, calibracao.json, camera_erro_px.csv) são as de `../jogo1/`.
