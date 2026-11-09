# Prática 3 — Futebol Analytics com YOLO

**Visão Computacional · CESAR School · 2026.2**

**Grupo:** Alana Costa, Henrique Bouwman e Thiago Alves

Pipeline que parte de um trecho de transmissão de **Manchester United × Roma (UEFA Champions League, 03/11/2026)**: detecta, rastreia e projeta jogadores e bola no campo, em metros. Com isso, calcula posse, passes, distância, eXP (*expected pass*) e métricas táticas extras.

> Os números deste README são **copiados dos arquivos gerados** em `saidas/` (ver a coluna "arquivo" de cada tabela). O README é regenerado por `src/preencher_readme.py` a partir desses arquivos.

---

## 0. Resumo

| Etapa | Resultado | Arquivo |
|---|---|---|
| Trecho | 2 min 50 s de câmera principal (relógio do jogo 2:26 → 5:16), 1280×720, processado a 15 fps (2.553 quadros) | `saidas/jogo1/meta.json` |
| Detecção | YOLOv8s (COCO), imgsz 1280: ⟪PESSOAS_Q⟫ pessoas/quadro, bola detectada em ⟪BOLA_DET⟫% dos quadros | `tracks.csv` |
| Bola + Kalman | YOLO ⟪BOLA_DET⟫% dos quadros → ⟪BOLA_DET_FILT⟫% após os filtros (gramado, tamanho, fora do tronco) → ⟪BOLA_FINAL⟫% com Kalman (⟪BOLA_KALMAN⟫ quadros por predição) | `bola.csv` |
| Times | K-Means k=3 (time 0 = vermelho/United, time 1 = cinza/Roma, outros = árbitro/goleiros) | `meta.json → times_bgr` |
| Homografia | calibração do `frame0` com erro ≤ 0,13 m nas linhas; o rastreamento da câmera só fica **verificado (≤ 3 px) em ⟪HOMOG_VALIDA⟫% dos quadros** | `calibracao.json`, `camera_erro_px.csv`, `homografia_validade.csv` |
| Posse (pixels, clipe inteiro) | United ⟪PX_POSSE_T0⟫% × Roma ⟪PX_POSSE_T1⟫% | `saidas/jogo1_px/metricas_times.json` |
| Passes detectados (pixels) | ⟪PX_PASSES_DET⟫ (United ⟪PX_PASSES_T0⟫, Roma ⟪PX_PASSES_T1⟫) | `saidas/jogo1_px/passes.csv` |
| Validação dos passes | 54 anotados à mão; P = ⟪PX_VAL_P⟫, R = ⟪PX_VAL_R⟫, F1 = ⟪PX_VAL_F1⟫ (pixels, tolerância 1 s) | `saidas/jogo1_px/validacao_passes.json` |
| Métricas em metros e eXP | só nos quadros verificados: demonstração do método, **não** conclusões | `saidas/jogo1/` |

> **Principal achado crítico.** A detecção, o rastreamento e a bola funcionam no clipe inteiro. O elo fraco é a **homografia ao longo do tempo**: a câmera de transmissão cruza o campo inteiro várias vezes em 3 min, e nossa cadeia "ECC + re-ancoragem por linhas" se perde e às vezes trava em linhas erradas.
>
> Por isso, as métricas principais (posse e passes) foram calculadas **em pixels** no clipe inteiro, alternativa prevista no roteiro. As métricas em metros (distância, compactação, Voronoi, eXP) foram calculadas **só nos quadros em que a reprojeção das linhas foi verificada**.

---

## 1. Vídeo e trecho (Parte 1)

- **Jogo:** Manchester United × Roma, UEFA Champions League 2026/27, 03/11/2026, Old Trafford. Transmissão CBS Sports.
- **Origem do arquivo:** página do jogo completo em <https://fullmatchsports.cc/manchester-united-vs-roma-full-match-ucl-2026-27/?tab=links>. O arquivo baixado tem 1920×1080 a 60 fps e 1h45min.
- **Intervalo usado:** **0:07:12,6 → 0:10:02,8 do arquivo** (170,2 s), que corresponde ao relógio do jogo **2:26 → 5:16** do 1º tempo.
- **O vídeo NÃO está no repositório.** Ele foi enviado pelo Google Drive: https://drive.google.com/file/d/1wBtYCyIeyYZclIhtWUh9rheJKp9TbgYC/view?usp=drive_link.

**Pré-processamento:**

```bash
ffmpeg -ss 432.6 -i mun.v.rom.full.ucl.03.11.2026.fullmatchsports.com.mp4 -t 170.2 \
       -vf "fps=30,scale=1280:720" -c:v libx264 -crf 20 -an mun_rom_0713_1003.mp4
ffmpeg -i mun_rom_0713_1003.mp4 -vf fps=15 -c:v libx264 -crf 18 clip15.mp4   # usado no pipeline
```

**Q1.1 — Por que este jogo e este trecho.** Testamos antes um PSG × Bayern, mas os trechos tinham muitos replays e closes. No Man United × Roma, analisamos o jogo inteiro: detectamos cortes de câmera nos quadros-chave e medimos a fração de gramado em cada quadro. O trecho escolhido é o mais longo de câmera principal com jogo corrido.

- O relógio do placar avança junto com o vídeo, então não há replay no trecho.
- Os uniformes contrastam bem: vermelho (United) × cinza-escuro com detalhes laranja (Roma). O árbitro usa ciano e o goleiro do United, verde-limão.
- A iluminação é noturna e uniforme.
- **Limitação:** há **um único close-up ao vivo** de outra câmera, de ~2,5 s (quadros 339–375 a 15 fps, ≈ 22,6–25,0 s). Esses quadros são excluídos das métricas em metros.
- O trecho tem 2 min 50 s, um pouco menos que os 3 min pedidos. Para chegar a 3 min teríamos que incluir closes de jogadores (Dalot e o goleiro, em 7:07–7:12).

**Q1.2 — Resolução, fps e custo.**

| Item | Valor |
|---|---|
| Original | 1920×1080 @ 60 fps |
| Clipe | 1280×720 @ 30 fps, 5.106 quadros |
| Processado | 1280×720 @ **15 fps, 2.553 quadros** |
| Hardware | só CPU (sem GPU): container na nuvem com 2 vCPU e VM local com 4 vCPU |
| Custo YOLOv8s, imgsz 1280 | ≈ 0,66 s/quadro (4 vCPU) a 1,3 s/quadro (2 vCPU) |

Processar os 5.106 quadros a 30 fps levaria ~1h50 numa máquina só.

- Por isso reduzimos para 15 fps.
- Também separamos a **detecção** (`detectar_cache.py`, em blocos retomáveis) do **rastreamento** (`rastrear_cache.py`, offline). A detecção foi dividida entre as duas máquinas: ~15 min de relógio.
- Essa separação também protegeu o trabalho: o container reiniciou no meio da primeira execução, e os blocos já detectados não se perderam.

---

## 2. Detecção com YOLO (Parte 2)

**Comparação de modelos (Q2.1).** Mesmo subtrecho: os primeiros 30 s do clipe de 30 fps, 1 quadro a cada 2 (450 quadros), em CPU. Arquivo: `saidas/comparacao_modelos/comparacao_modelos.csv`, gerado por `src/comparar_modelos.py`.

⟪TABELA_MODELOS⟫

- **O que mais pesa é a resolução de entrada, não o tamanho do modelo.** Em 640 px, a bola aparece em só 33–55% dos quadros. Em 1280 px, chega a 91–94% com qualquer tamanho (n, s ou m).
- Para jogadores, a diferença entre os modelos é pequena: de 12,1 a 13,6 por quadro.
- Em 1280 px, porém, o detector devolve **~2 "bolas" por quadro**, a maioria falsos positivos.
- Escolhemos o **YOLOv8s @ 1280**. Ele detecta a bola quase tão bem quanto o `m` e é 2,7× mais rápido. O `n` @ 1280 seria uma alternativa ainda mais rápida, com resultado parecido.

**Q2.2 — Por que a bola é difícil.**
- Ela tem ~6–10 px de diâmetro em 1280×720. Em 640 px, cai para 3–5 px, menor que a resolução útil das camadas mais profundas do detector.
- Em passes e chutes há borrão de movimento, e ela fica oculta pelos pés.
- No COCO, `sports ball` inclui bolas de vários esportes, em close.
- O resultado bate com Ahmad Yousef et al. (2025, Tabela 2): F1 da bola ≈ 0,31 sem filtro temporal contra ≈ 0,96 com Kalman.

**Q2.3 — `imgsz` e `conf-bola`.**
- O `imgsz` é o fator principal, como mostra a tabela.
- Com `conf-bola = 0,10`, aparecem muitos falsos positivos. A figura `figuras/bola_falsos_positivos.png` mostra candidatos rejeitados pelo *gating* do Kalman: o goleiro reserva de amarelo na arquibancada, objetos brancos e, durante o close-up, a própria bola em escala gigante.
- ⟪BOLA_FP_PCT⟫% dos candidatos ficam a mais de 40 px da bola aceita.

---

## 3. Rastreamento e times (Parte 3)

**Q3.1 — Trocas de ID.** Na versão final (`src/rastrear_v2.py`) usamos o **BoT-SORT com compensação de movimento de câmera** (GMC por fluxo óptico esparso), em vez do ByteTrack puro. Com os *pans* rápidos da transmissão, o ByteTrack associava as caixas por IoU sem descontar o movimento da câmera.

Mesmo cache de detecções, sem vídeo:

| Rastreador | IDs em 170 s |
|---|---|
| ByteTrack | 626 |
| BoT-SORT + GMC | 525 |
| BoT-SORT + GMC com buffer de 4 s | 512 |

Ficamos com o buffer de 2 s. O resultado final: **⟪N_TRACKS⟫ IDs** com ~14 pessoas visíveis por quadro, ou seja, **~⟪IDS_MIN⟫ IDs novos por minuto**. Ainda é muito.

As trocas acontecem principalmente:
1. quando jogadores saem e voltam ao quadro nos *pans* rápidos (a câmera cruza o campo inteiro várias vezes);
2. em disputas e aglomerações (oclusão);
3. no close-up;
4. por causa dos 15 fps: o deslocamento entre quadros dobra e a associação por IoU falha mais.

Consequência: distância percorrida "por jogador" fica fragmentada entre vários IDs. Por isso as métricas físicas são reportadas por **time**, não por jogador.

**Q3.2 — K-Means de cor.** Mudamos duas coisas em relação ao `detectar_rastrear.py` original. As duas resolveram erros que vimos no `rastreado.mp4`:
1. **K-Means ajustado com amostras do clipe inteiro** (6.000 detecções), não só dos 60 primeiros quadros. No original, os primeiros 4 s (60 quadros a 15 fps) ficavam **sem time** (`?`), porque o classificador ainda estava "calibrando".
2. **Time por track com moda móvel** (janela de 15 detecções) **no lugar do voto fixo**. No original, o time de um track ficava travado depois de 10 votos. Quando o ID "pulava" de um jogador do United para um da Roma (troca de ID numa disputa), o segundo jogador herdava o time errado. Com a moda móvel, o rótulo muda no ponto da troca.

Os centróides estão em `meta.json → times_bgr`, e há recortes de exemplo em `figuras/times_cores.png`:

| Grupo | Cor média (BGR) | Quem é |
|---|---|---|
| Time 0 | ⟪COR_T0⟫ | Manchester United (vermelho) |
| Time 1 | ⟪COR_T1⟫ | Roma (cinza-escuro) |
| Outros (−2) | ⟪COR_OUT⟫ | árbitros (ciano) e goleiros |

O classificador usa (S·cos H, S·sin H, V) porque **o matiz é um ângulo**. O vermelho fica nas duas pontas da escala (H ≈ 0 e H ≈ 179). A média simples de H de uma camisa vermelha daria ≈ 90 (ciano), justamente a cor do árbitro. Com seno e cosseno, a média de matizes vermelhos continua vermelha. Neste jogo, isso é exatamente o que separa o United do árbitro.

**Erros observados:**
- jogadores do United em sombra ou com borrão às vezes caem em "outros" (há um exemplo em `times_cores.png`);
- **goleiros classificados no time errado:** o goleiro do United (verde-limão) aparecia como **Roma** no `rastreado.mp4`. O filtro de grama (H ∈ [35, 85]) apaga quase toda a camisa dele, e sobram calção e meião escuros, que caem no cluster cinza.

  **Correção (`src/corrigir_goleiros.py`):** para cada track, medimos a fração de pixels verde-limão/amarelo vivo na camisa (H ∈ [25, 50], S > 90, V > 170; o gramado noturno é bem mais escuro). Os tracks acima de 20% são separados em dois papéis:
  - **goleiro:** tem gramado sob os pés. O time sai do matiz: verde-limão (H ≥ 36) = United (time 0); amarelo (H < 36) = Roma (time 1). Goleiro **entra no time**, porque participa de posse e passes (tiro de meta, reposição). No vídeo aparece como `GK0#id` / `GK1#id`.
  - **staff:** está fora do gramado (pista/placas), como o funcionário de colete amarelo. Vai para "outros" e aparece como `Staff#id`. O árbitro continua no 3º cluster do K-Means, como `Arb#id`.

  A lista conferida visualmente está em `saidas/jogo1/goleiros.csv`: ⟪N_GK⟫ tracks. ⟪N_GK_ROMA⟫ deles estavam como Roma antes da correção. O `tracks.csv` original ficou em `tracks_antes_goleiro.csv`, e o `rastreado.mp4` é redesenhado com `rastrear_v2.py --goleiros`.
- A lição para o K-Means de cor: uniforme cuja cor se parece com o gramado é um ponto cego do filtro de grama. A cor sozinha não basta para separar goleiros; um modelo com a classe *goalkeeper* (Q2.4) resolveria.

**Q3.3 — `k=3`.** Usamos `k=3` desde o início.
- **Vantagem:** o árbitro, de ciano, ficou isolado em "outros". Assim ele não "rouba" posse nem vira receptor de passe.
- **Risco:** se o 3º cluster não for o árbitro, ele pode partir um dos times (por exemplo, o vermelho em sombra). Em trechos sem árbitro visível no início, o 3º cluster pode ficar sem sentido.

---

## 4. Bola e homografia (Parte 4)

**Q4.1 — Kalman.**

| Item | Valor |
|---|---|
| Detecção crua | bola em ⟪BOLA_DET⟫% dos quadros (⟪BOLA_CAND⟫ candidatos) |
| Após os filtros (gramado, tamanho, fora do tronco) | bola em ⟪BOLA_DET_FILT⟫% dos quadros |
| Após o *gating* (Mahalanobis) | ⟪BOLA_ACEITA⟫ detecções aceitas |
| Predição do Kalman | ⟪BOLA_KALMAN⟫ quadros preenchidos |
| Cobertura final | ⟪BOLA_FINAL⟫% |

A linha do tempo está em `figuras/bola_cobertura.png`.
- **Onde o Kalman acerta:** lacunas curtas durante passes rasteiros.
- **Onde ele travou errado (e corrigimos):** no início do clipe, o Kalman aceitou falsos positivos na torcida e nas placas (y ≈ 50–100 px) e chegou a prever a bola fora da imagem. Resultado: ninguém ficava "dono" da bola por 29 s. Agora o `rastrear_v2.py` filtra os candidatos **antes** do Kalman. Isso eliminou ⟪BOLA_REMOV⟫ candidatos. Fica só quem:
- tem gramado em volta (anel de 25 px);
- está fora da faixa do placar;
- tem tamanho de bola (4–30 px, quase quadrado);
- **não está no tronco ou na cabeça de um jogador** (cabeças carecas e meiões brancos eram falsos positivos frequentes).

Depois do Kalman, também removemos predições fora da imagem.
- **Parâmetros do Kalman ajustados ao fps:** os padrões do professor (`--max-lacuna 15`, `--gate 6`) são para 25–30 fps. A 15 fps, 15 quadros são 1 s de bola "inventada", e o deslocamento entre quadros dobra, então o *gate* rejeitava a bola verdadeira depois de passes rápidos. Usamos `--max-lacuna 8` (≈ 0,5 s, o mesmo tempo do padrão a 30 fps) e `--gate 9`.

  Testamos outras combinações contra a anotação manual, e o F1 dos passes variou de 0,41 a 0,59. Para não "decorar o gabarito", conferimos a escolha **separadamente em cada metade do clipe**. Com o padrão (15, 6), o F1 foi 0,46 na 1ª metade e 0,53 na 2ª. Com (8, 9), foi 0,58 e 0,59. A melhora aparece nas duas metades, mas os números finais de validação continuam um pouco otimistas.
- **Onde ele "inventa":** quando a bola é chutada alto ou sai do quadro, a predição de velocidade constante continua em linha reta por até 8 quadros (≈ 0,5 s). No close-up (22,6–25 s), ele segue uma bola que não está mais na imagem.

**Q4.2 — Calibração.**
- No `frame0` a câmera está fechada na grande área direita. Os quatro marcos "clicáveis" ficam todos numa faixa estreita da imagem.
- Com eles, a H0 extrapolava muito mal: a linha lateral projetada saía ~15 m fora.
- Por isso calibramos por **ajuste de linhas** (`src/calibrar_linhas.py`): mínimos quadrados robustos sobre os pixels de 5 linhas do campo (linha de fundo, frente e lateral da grande área, lateral do campo e frente da pequena área). A inicialização usa as interseções dessas linhas, inclusive fora da imagem.

Resíduo por linha, em `calibracao.json → rms_linhas_m`:

| Linha | RMS (m) |
|---|---|
| Linha de fundo (x = 105) | ⟪RMS_X105⟫ |
| Frente da grande área (x = 88,5) | ⟪RMS_X885⟫ |
| Lateral da grande área (y = 54,16) | ⟪RMS_Y5416⟫ |
| Lateral do campo (y = 68) | ⟪RMS_Y68⟫ |
| Frente da pequena área (x = 99,5) | ⟪RMS_X995⟫ |

Erro de reprojeção dos marcos clicados:

| Marco | Erro (m) |
|---|---|
| ga_dir_sup | ⟪E_GA_SUP⟫ |
| ga_dir_inf | ⟪E_GA_INF⟫ |
| ga_dir_inf_linha | ⟪E_GA_INF_L⟫ |
| pa_dir_inf | ⟪E_PA_INF⟫ |

- O marco com mais erro é a **quina superior (mais distante) da grande área**: ali 1 px corresponde a vários centímetros de campo, e o clique é impreciso.
- **Validação independente:** a meia-lua, que não entrou no ajuste, cai sobre a linha real (`figuras/calibracao_linhas_campo.jpg`). A grade de 5 m está em `figuras/calibracao_grade.jpg`.

**Q4.3 — `H0·G_t`, deriva e re-ancoragem.**
- A posição no campo é `[x, y, 1]ᵀ ~ H0 · G_t · [u, v, 1]ᵀ`, onde `G_t` leva os pixels do quadro t para o quadro 0.
- O `homografia.py camera` original (ORB + quadros-chave) **divergiu em ~7 s** neste clipe. Os pontos ORB caem nas placas de LED, que são animadas, e o gramado quase não tem textura. A câmera atravessa o campo inteiro várias vezes no trecho.

**Nossa proposta (`src/camera_linhas.py`):**
1. Movimento relativo entre quadros por **ECC** (homografia), inicializado por **correlação de fase**, usando só o gramado (sem jogadores e sem placar).
2. **Re-ancoragem a cada 5 quadros:** reprojeta todas as linhas oficiais do campo (incluindo o círculo central), coleta os pixels de linha a até ±6 px e reajusta `H_t`. O reajuste só é aceito se houver linhas nas duas direções e resíduo mediano ≤ 0,35 m. Isso zera a deriva.
3. **Relocalização quando se perde:** testa como hipóteses 2 calibrações-semente (`src/sementes_camera.py`: área esquerda e meio-campo) e as últimas poses aceitas, com deslocamentos horizontais.
4. **Verificação quadro a quadro** (`src/checar_camera.py --passo-check 1`): para cada quadro, medimos a distância mediana entre as linhas reprojetadas e os pixels de linha detectados. Só os quadros com **≤ 3 px** (e fora do close-up) entram nas métricas em metros (`src/filtrar_projecao.py`).

**Resultado honesto:**
- Ancoragens aceitas: ⟪ANC_ACEITAS⟫ de ⟪ANC_TOTAL⟫.
- **Quadros verificados: ⟪HOMOG_VALIDA⟫%.** Erro mediano em todo o clipe: ⟪CAM_ERRO_MED⟫ px.
- O mosaico `figuras/camera_reprojecao.jpg` mostra o padrão. A projeção é perfeita no início (área direita, 0–6 s) e logo após as sementes (≈ 20–23 s e 29–34 s). Nos *pans* rápidos de uma área à outra, o ECC perde o alinhamento.
- Pior: a re-ancoragem às vezes **trava em linhas erradas** com resíduo baixo. Linhas do campo são ambíguas: a frente da grande área pode "casar" com a da pequena área, e a lateral com a linha de fundo. Por isso a aceitação da ancoragem sozinha não basta e precisamos da verificação do item 4.

**O que faríamos com mais tempo:** calibração **por quadro**, sem cadeia, usando um detector de pontos-chave do campo (como os modelos de *pitch keypoints* do Roboflow ou o desafio de calibração do SoccerNet). Isso elimina a deriva por construção. Com ele, cada quadro tem sua H própria, e cortes e *pans* deixam de ser um problema.

**Q4.4 — Pé × centro.**
- A homografia só vale para pontos **no plano do gramado**. O pé (base do box) está no chão; o centro do box está ~0,9 m acima. Projetar o centro deslocaria o jogador para "trás", e o erro cresce com a distância à câmera.
- Para a bola no ar (lançamentos, cruzamentos), nem o centro nem a base estão no plano: a projeção a coloca longe do ponto real. Seria preciso estimar a altura, por exemplo com a sombra ou um modelo balístico com a trajetória do Kalman.

---

## 5. Métricas de jogo (Parte 5)

**Escolha de unidade.** Como a homografia só foi verificada em ⟪HOMOG_VALIDA⟫% dos quadros, a posse e os passes do **clipe inteiro** foram calculados **em pixels** (`metricas.py --pixels`: raio de posse 35 px ≈ 1,5–3 m, conforme a profundidade na imagem). O que depende de metros foi calculado só nos quadros verificados.

**Posse indefinida.** No script original, a posse é "herdada" pelo último dono até o rival tocar na bola. Quando a bola não é detectada por vários segundos (bola no ar, *pans* rápidos), isso dava 10+ s de posse a um time sem evidência: no início do vídeo, 100% Roma até o segundo 29, mesmo com o United tendo a bola. Adicionamos `--max-lacuna-posse 30`: sem dono observado por mais de 2 s, a posse fica **indefinida**. A posse (%) é calculada sobre os quadros definidos; indefinidos: ⟪PX_INDEF⟫% dos quadros.

**Posse e passes, em pixels, no clipe inteiro** (`saidas/jogo1_px/metricas_times.json`):

| Métrica | United (T0) | Roma (T1) |
|---|---|---|
| Posse (%) | ⟪PX_POSSE_T0⟫ | ⟪PX_POSSE_T1⟫ |
| Passes detectados | ⟪PX_PASSES_T0⟫ | ⟪PX_PASSES_T1⟫ |
| Passes certos | ⟪PX_CERTOS_T0⟫ | ⟪PX_CERTOS_T1⟫ |
| Precisão (%) | ⟪PX_PREC_T0⟫ | ⟪PX_PREC_T1⟫ |
| **Anotação manual: passes (certos)** | **37 (29)** | **17 (10)** |

**Métricas em metros, só nos quadros verificados** (`saidas/jogo1/metricas_times.json`, ⟪M_QUADROS_BOLA⟫ quadros com bola projetada):

| Métrica | United (T0) | Roma (T1) |
|---|---|---|
| Posse (%) | ⟪POSSE_T0⟫ | ⟪POSSE_T1⟫ |
| Passes detectados | ⟪PASSES_T0⟫ | ⟪PASSES_T1⟫ |
| Distância total, soma dos IDs (m) | ⟪DIST_T0⟫ | ⟪DIST_T1⟫ |

**Q5.1 — Plausibilidade.**
- Nos quadros verificados, a velocidade máxima foi de ⟪VMAX⟫ m/s. A média de todos os IDs foi de ⟪M_POR_MIN⟫ m/min, contra a referência de ~120 m/min.
- Os valores **não são fisicamente plausíveis como médias do jogo**, e entendemos por quê:
- só ⟪HOMOG_VALIDA⟫% dos quadros têm homografia verificada, e eles se concentram em poucos trechos com muita movimentação perto da área;
- os IDs são curtos (trocas frequentes), então cada "jogador" é observado por poucos segundos;
- o resíduo de projeção (alguns px viram dezenas de cm longe da câmera) soma deslocamento falso a cada quadro, mesmo com a mediana móvel de 5 quadros.

A velocidade máxima (≈ 10 m/s, 36 km/h) está no limite humano de um sprint. O `metricas.py` descarta saltos acima de 12 m/s.

**Q5.2 — Sensibilidade ao raio de posse** (pixels, clipe inteiro; `saidas/jogo1_px/sensibilidade_raio.csv` e `figuras/sensibilidade_raio_px.png`):

⟪TABELA_RAIO_PX⟫

A **posse é estável**: o United fica entre 67% e 69% em todos os raios de 20 a 80 px. Já o **número de passes é sensível**: vai de 34 a 59.
- Com raio pequeno, só ~23% dos quadros têm "dono", e passes com pouco contato são perdidos.
- Com raio grande, a bola é atribuída a quem só passa perto, o que gera passes "falsos" entre jogadores próximos. A precisão dos passes do United cai de 74% para 56%.

Comparamos com a anotação manual (tolerância 1 s): F1 de 0,52 (20 px), **0,59 (35 px)**, 0,51 (50 px) e 0,44 (65 px). O padrão do professor, 35 px, foi o melhor e foi mantido.

**Q5.3 — Validação manual.**
- Anotamos à mão **54 passes** no vídeo de 30 fps (United 37, sendo 29 certos; Roma 17, sendo 10 certos). Arquivo: `saidas/jogo1/passes_anotados.csv`. O original, como exportado do Excel, está em `passes_anotados_original.csv`.
- Um passe detectado "casa" com um anotado se for do mesmo time e estiver a até 1 s de distância (`validar_passes.py`).

| Versão | Tolerância | Detectados | Anotados | Casados | Precisão | Revocação | F1 | Acordo certo/errado |
|---|---|---|---|---|---|---|---|---|
⟪TABELA_VALIDACAO⟫

Na versão final, o pipeline detecta **⟪PX_PASSES_DET⟫ passes** e nós anotamos 54. Casam ⟪PX_CASADOS⟫: precisão ⟪PX_VAL_P⟫, revocação ⟪PX_VAL_R⟫ e **F1 ⟪PX_VAL_F1⟫**. Com tolerância de 2 s, o F1 sobe para ⟪PX_VAL_F1_2⟫.

Por time, a detecção fica perto da anotação:
- **United:** ⟪PX_PASSES_T0⟫ passes detectados (⟪PX_PREC_T0⟫% certos), contra 37 anotados (78% certos).
- **Roma:** ⟪PX_PASSES_T1⟫ detectados (⟪PX_PREC_T1⟫% certos), contra 17 anotados (59% certos).

**Os erros que restam são equilibrados entre os dois tipos:**
- **Falsos negativos:** passes de primeira (a bola fica menos de 3 quadros, 0,2 s, perto do jogador); trocas de ID do passador ou do receptor; lançamentos com a bola no ar ou fora do quadro.
- **Falsos positivos:** bola passando perto de um adversário vira "passe errado" (por isso a precisão da Roma fica abaixo da real); trocas de ID do mesmo jogador.

**Como chegamos aqui.** Cada passo foi motivado pelo que vimos no `rastreado.mp4` e medido contra a anotação, com tolerância de 1 s:

| Versão | O que mudou | Passes detectados | P | R | F1 |
|---|---|---|---|---|---|
| v1 | pipeline do professor (ByteTrack, times dos 60 primeiros quadros, bola sem filtro) | 23 | 0,70 | 0,30 | 0,42 |
| v2 | + goleiros e staff | 22 | 0,73 | 0,30 | 0,42 |
| v3 | + filtro de bola no gramado e posse indefinida após 2 s | 43 | 0,49 | 0,39 | 0,43 |
| **final** | + `rastrear_v2.py` (K-Means no clipe inteiro, moda móvel de time, BoT-SORT + GMC, filtro de tamanho/tronco da bola) e Kalman ajustado a 15 fps | **⟪PX_PASSES_DET⟫** | **⟪PX_VAL_P⟫** | **⟪PX_VAL_R⟫** | **⟪PX_VAL_F1⟫** |

As versões intermediárias não estão versionadas em `saidas/`; só a final está. Os números delas vêm das nossas execuções durante o trabalho.

**Ressalva:** os parâmetros do Kalman foram escolhidos olhando esta mesma anotação (ver Q4.1), então o F1 final é um pouco otimista.

Quando o passe é encontrado, o rótulo certo/errado bate com a anotação em ⟪PX_ACORDO⟫ dos casos.

A versão em metros quase não detecta passes, porque só 8% dos quadros entram.

**Q5.4 — Direção de ataque e efeito das trocas de ID.**
- A heurística automática usa os quantis extremos de x de cada time. Ela só funciona quando os dois goleiros aparecem, e aqui a câmera mostra no máximo metade do campo por vez. Por isso forçamos a direção.
- **Trocas de ID:** a distância por jogador fica fragmentada. Um passe pode aparecer como "troca de dono" quando é só o mesmo jogador com um ID novo (o filtro de distância mínima de 3 m reduz isso).
- **Jogadores fora do quadro:** não são contados, o que subestima a distância do time e a ocupação do campo.
- **Correção proposta:** re-identificação por aparência (número da camisa ou embeddings), ou fundir tracks do mesmo time quando um some e outro aparece perto em menos de ~1 s.

---

## 6. eXP — expected pass (Parte 6)

**Q6.1 — Features (`passes_features.csv`) e relação com os 17 canais de Gomes et al. (2026, Tabela 1):**

| Feature | O que mede | Canal(is) análogo(s) na SoccerMap |
|---|---|---|
| `distancia` | comprimento do passe | distância bola → célula |
| `adversarios_na_linha` | adversários perto da linha do passe | densidade de defensores, ângulo/linha de passe |
| `pressao_passador` | adversários perto do passador | localização e densidade de defensores perto da bola |
| `espaco_destino` | distância do destino ao adversário mais próximo | densidade de defensores na célula de destino |
| `progressao_x` | avanço em direção ao gol | coordenada x da célula |
| `dist_destino_gol` | distância do destino ao gol | distância/ângulo da célula ao gol |

**O que não conseguimos extrair do vídeo:**
- **velocidades** confiáveis de todos os jogadores (os canais de velocidade): IDs trocam e a câmera corta jogadores;
- **todos os 22 jogadores** (a câmera mostra metade do campo);
- a **altura da bola**;
- a **identidade do jogador e do goleiro** de forma estável.

**Q6.2 — Comparação dos modelos.**

**Não foi possível avaliar o eXP com os dados deste trecho.**
- Nos quadros com homografia verificada (≈ 8%), sobrou só ⟪N_PASSES_M⟫ passe em metros, e o `xpass.py` só consegue rodar o heurístico.
- Uma execução anterior, com um filtro mais permissivo (só a aceitação da ancoragem), produziu 16 passes. Descartamos esse resultado, e por isso ele não está em `saidas/`, porque muitas posições estavam projetadas com a homografia errada: as AUCs variavam entre os modelos sem nenhum padrão.

Isso mostra o ponto da questão: **com poucos passes, AUC, Brier e Log-Loss são instáveis.** Com 16 passes, trocar 1 rótulo muda a AUC em ~0,1. O logístico com validação cruzada fica pior que o baseline por sobreajuste.

Para uma avaliação séria seria preciso juntar passes de outros trechos ou grupos (`--extra`). Como referência, Gomes et al. treinam com milhares de passes e chegam a AUC de 0,622.

**Q6.3 — Calibração.** Sem passes suficientes, não há curva de calibração confiável, então ela não foi gerada. Pelo desenho, o modelo heurístico (coeficientes fixos, intercepto 2,5) tende a prever probabilidades altas. Só uma curva de calibração com dezenas de passes mostraria se ele é otimista demais.

**Q6.4 — Superfície de eXP.** A superfície gerada num quadro verificado (`figuras/superficie_xp_f470.png`, meio-campo) é coerente com a intuição tática:
- **regiões quentes:** perto do passador e nos espaços sem adversários próximos;
- **regiões frias:** atrás da linha de defensores e longe da bola.

É o mesmo comportamento da Fig. 1 de Gomes et al., mas com só 6 features no lugar de 17 canais.

**Q6.5 — What-if e PAx** (`what_if.csv`, `jogadores_xp.csv`): Com 1 passe em metros, o *what-if* e o PAx não têm valor estatístico. O exemplo mostra só a mecânica: deslocar o destino 3–6 m e recalcular o eXP. Mesmo com dados completos, 3 minutos dão 1–3 passes por jogador, e o PAx por jogador seria ruído. Ele só faz sentido agregado em vários jogos.

---

## 7. Métricas extras (proposta do grupo)

Calculadas em `src/metricas_extras.py` sobre os jogadores **visíveis**, **só nos quadros com homografia verificada**, amostrando 1 quadro a cada 5. Arquivos: `metricas_extras.csv` e `metricas_extras.json`.

| Métrica | United | Roma | Para que serve |
|---|---|---|---|
| Jogadores visíveis (média) | ⟪EX_N_T0⟫ | ⟪EX_N_T1⟫ | — |
| Largura do bloco P90−P10 (m) | ⟪EX_LARG_T0⟫ | ⟪EX_LARG_T1⟫ | amplitude ofensiva × fechamento defensivo |
| Profundidade do bloco P90−P10 (m) | ⟪EX_PROF_T0⟫ | ⟪EX_PROF_T1⟫ | compactação vertical |
| Área do fecho convexo (m²) | ⟪EX_AREA_T0⟫ | ⟪EX_AREA_T1⟫ | compactação; cai quando o time defende |
| Linha defensiva média, x (m) | ⟪EX_LINHA_T0⟫ | ⟪EX_LINHA_T1⟫ | altura do bloco, pressão alta × bloco baixo |
| Controle de espaço Voronoi (%) | ⟪EX_VOR_T0⟫ | ⟪EX_VOR_T1⟫ | domínio territorial (*pitch control* simplificado) |
| Adversários a ≤ 5 m do portador | ⟪EX_PRES_T0⟫ | ⟪EX_PRES_T1⟫ | intensidade da pressão sofrida |

Figuras: `figuras/extras_series.png` (séries no tempo) e `figuras/voronoi_exemplo.png`.

**Outras métricas que daria para extrair com o mesmo pipeline:**
- *pitch control* físico, com o tempo de chegada a cada célula (Spearman);
- passes progressivos e passes que "quebram linhas" (*packing*);
- *expected threat* (xT) para valorar conduções;
- mapa de pressão e PPDA, aproximado pelas ações defensivas;
- sprints acima de 7 m/s (exige IDs estáveis).

**Leitura (com cautela, ⟪EX_QUADROS⟫ quadros verificados):**
- a área do bloco fica em ~570 m² para o United e ~450 m² para a Roma, com ~5–7 jogadores visíveis de cada time;
- o controle Voronoi fica em 57% × 43% a favor do United, coerente com a posse;
- a "linha defensiva" (os 3 jogadores visíveis mais recuados) fica em x ≈ 76 m para o United (que defende x = 105) e x ≈ 57 m para a Roma (que defende x = 0). Para a Roma, isso mostra a limitação: a câmera quase nunca mostra seus zagueiros quando a bola está no ataque dela, então a métrica vira a "linha mais recuada **visível**".

São métricas baratas de calcular com o pipeline e úteis para comparar fases do jogo (pressão alta × bloco baixo). Mas exigem homografia confiável e, idealmente, os 22 jogadores (câmera tática).

---

## 8. Limitações principais

1. **Só CPU:** 15 fps e YOLOv8s COCO, sem *fine-tuning* com classes *player/goalkeeper/referee/ball*. O Q2.4 opcional não foi feito.
2. **Câmera de transmissão:** só ~metade do campo aparece; há *pans* rápidos e um close-up. As métricas são "do que se vê".
3. **Muitas trocas de ID:** distância e PAx por jogador não são confiáveis.
4. **Poucos passes (~50 anotados):** AUC, Brier e Log-Loss do eXP têm intervalos de confiança largos. É uma demonstração do método, não uma avaliação conclusiva.

---

## 9. Como reproduzir

```bash
pip install -r requirements.txt
cd src
S=../saidas/jogo1
# 1) detecção em blocos (retomável; pode rodar em várias máquinas com --ini/--fim)
python detectar_cache.py --video ../dados/clip15.mp4 --saida $S --modelo yolov8s.pt --imgsz 1280
# 2) rastreamento + times
python rastrear_v2.py --video ../dados/clip15.mp4 --saida $S --sem-video
python corrigir_goleiros.py --video ../dados/clip15.mp4 --saida $S --limiar 0.2   # goleiros → seu time; staff → outros
python rastrear_v2.py --video ../dados/clip15.mp4 --saida $S --goleiros $S/goleiros.csv   # tracks finais + rastreado.mp4
python bola_kalman.py --saida $S --max-lacuna 8 --gate 9
python filtrar_bola_gramado.py --video ../dados/clip15.mp4 --saida $S --limpar-bola
python calibrar_linhas.py --saida $S
python sementes_camera.py --video ../dados/clip15.mp4 --saida $S
python camera_linhas.py --saida $S
python homografia.py projetar --saida $S
python filtrar_projecao.py --saida $S
python checar_camera.py --saida $S
# 4) métricas, validação, eXP e extras
python metricas.py --saida $S --ataque-time0 esquerda --max-lacuna-posse 30
python validar_passes.py --saida $S --anotado $S/passes_anotados.csv --tolerancia 1
python validar_passes.py --saida $S --anotado $S/passes_anotados.csv --tolerancia 2
python sensibilidade_raio.py --saida $S
# versão em pixels (métricas principais): copie tracks.csv, bola.csv e meta.json para ../saidas/jogo1_px e rode
# python metricas.py --saida ../saidas/jogo1_px --pixels --ataque-time0 esquerda --max-lacuna-posse 30
# python sensibilidade_raio.py --saida ../saidas/jogo1_px --pixels --raios 20,35,50,65,80 --max-lacuna-posse 30
python metricas_extras.py --saida $S
python xpass.py --saida $S --what-if
python figuras_deteccao.py --saida $S
python video_radar.py --saida $S --inicio 0 --fim 120     # vídeo da apresentação (vai para o Drive)
# comparação de modelos (Parte 2)
python comparar_modelos.py ../dados/mun_rom_0713_1003.mp4 yolov8s.pt 1280 0 900
```

**Estrutura:**
- `src/`: scripts. Os do professor (`vc-20262-p3`) mais os nossos: `detectar_cache.py`, `rastrear_cache.py` (v1), `rastrear_v2.py`, `corrigir_goleiros.py`, `filtrar_bola_gramado.py`, `calibrar_linhas.py`, `sementes_camera.py`, `camera_linhas.py`, `filtrar_projecao.py`, `checar_camera.py`, `metricas_extras.py`, `sensibilidade_raio.py`, `figuras_deteccao.py`, `video_radar.py`, `comparar_modelos.py` e `preencher_readme.py`.
- `saidas/jogo1/`: CSVs e JSONs gerados, mais a anotação manual. Os vídeos ficam de fora.
- `saidas/comparacao_modelos/`: Parte 2.
- `figuras/`: todas as figuras.

---

## 10. Declaração de uso de IA

Usamos o **Claude (Anthropic)** como assistente de programação e de redação. Ele:
- adaptou e escreveu scripts (`detectar_cache.py`, `rastrear_cache.py` (v1), `rastrear_v2.py`, `corrigir_goleiros.py`, `filtrar_bola_gramado.py`, `calibrar_linhas.py`, `sementes_camera.py`, `camera_linhas.py`, `filtrar_projecao.py`, `checar_camera.py`, `metricas_extras.py`, `sensibilidade_raio.py`, `figuras_deteccao.py`, `video_radar.py`, `comparar_modelos.py`, `preencher_readme.py`);
- executou o pipeline;
- ajudou na redação inicial deste README e dos slides.

**O grupo:**
- escolheu o jogo e o trecho;
- fez a **anotação manual dos 54 passes**;
- conferiu visualmente o `rastreado.mp4` (foi assim que identificamos o goleiro do United classificado como Roma), a calibração e o mosaico de reprojeção;
- revisou e reescreveu a análise.

Os scripts de partida são do repositório da disciplina (`eron93br/vc-20262-p3`). [REVISAR: o grupo deve ajustar esta declaração ao que de fato fez.]
