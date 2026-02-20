"""
=============================================================
  ANÁLISE DE FATURAMENTO — ANALISTA DE DADOS JÚNIOR
  Base: base_faturamento__1_.csv
  Tópicos: Inadimplência, Consumo, Faturamento, Vencimentos,
           Atrasos, Clientes VIP, Frequência de atraso
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")

# ── Estilo global ──────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d0f14",
    "axes.facecolor":    "#161a23",
    "axes.edgecolor":    "#2a3045",
    "axes.labelcolor":   "#a0aec0",
    "axes.titlecolor":   "#e8ecf4",
    "xtick.color":       "#6b7a99",
    "ytick.color":       "#6b7a99",
    "grid.color":        "#2a3045",
    "text.color":        "#e8ecf4",
    "font.family":       "monospace",
    "axes.grid":         True,
    "grid.alpha":        0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

AMARELO  = "#f0c040"
TEAL     = "#4fd1c5"
VERMELHO = "#f87171"
VERDE    = "#4ade80"
ROXO     = "#a78bfa"
BRANCO   = "#e8ecf4"
MUTED    = "#6b7a99"
BG       = "#0d0f14"
SURFACE  = "#161a23"

# ══════════════════════════════════════════════════════════
# 0. CARREGAMENTO E LIMPEZA
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  ANÁLISE DE FATURAMENTO")
print("=" * 60)

df = pd.read_csv(
    "/mnt/user-data/uploads/base_faturamento__1_.csv",
    sep=";",
    decimal=","
)

df["valor_fatura"] = (
    df["valor_fatura"].astype(str).str.replace(",", ".").astype(float)
)
df["data_vencimento"] = pd.to_datetime(df["data_vencimento"], dayfirst=True)
df["dia_vencimento"]  = df["data_vencimento"].dt.day
df["mes_vencimento"]  = df["data_vencimento"].dt.to_period("M").astype(str)

print(f"\nRegistros carregados : {len(df)}")
print(f"Colunas              : {df.columns.tolist()}")
print(f"Competências         : {sorted(df['competencia'].unique())}")
print(f"Clientes únicos      : {df['id_cliente'].nunique()}")
print(f"Status possíveis     : {df['status_fatura'].unique()}")
print(f"\nValores ausentes:\n{df.isnull().sum()}")

# ══════════════════════════════════════════════════════════
# 1. TAXA DE INADIMPLÊNCIA
# ══════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  1. TAXA DE INADIMPLÊNCIA")
print("─" * 60)

status_counts = df["status_fatura"].value_counts()
total         = len(df)
tx_atrasada   = status_counts.get("atrasada", 0) / total * 100
tx_aberto     = status_counts.get("em aberto", 0) / total * 100
tx_paga       = status_counts.get("paga", 0) / total * 100
valor_atrasado = df.loc[df["status_fatura"] == "atrasada", "valor_fatura"].sum()
valor_total    = df["valor_fatura"].sum()

print(f"\n  Pagas      : {status_counts.get('paga',0):>4} ({tx_paga:.1f}%)")
print(f"  Atrasadas  : {status_counts.get('atrasada',0):>4} ({tx_atrasada:.1f}%)")
print(f"  Em aberto  : {status_counts.get('em aberto',0):>4} ({tx_aberto:.1f}%)")
print(f"\n  Valor total faturado : R$ {valor_total:>10,.2f}")
print(f"  Valor em atraso      : R$ {valor_atrasado:>10,.2f}")
print(f"  % do faturamento     : {valor_atrasado/valor_total*100:.1f}%")

# por competência
print("\n  Inadimplência por competência:")
for comp in sorted(df["competencia"].unique()):
    sub  = df[df["competencia"] == comp]
    atr  = (sub["status_fatura"] == "atrasada").sum()
    taxa = atr / len(sub) * 100
    print(f"    {comp}: {atr}/{len(sub)} atrasadas → {taxa:.1f}%")

# ══════════════════════════════════════════════════════════
# 2. COMPORTAMENTO DE CONSUMO
# ══════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  2. CONSUMO DE ENERGIA")
print("─" * 60)

print("\n  Por tipo de cliente:")
print(df.groupby("tipo_cliente")["consumo_energia_kwh"]
        .agg(media="mean", mediana="median", maximo="max", total="sum")
        .round(1).to_string())

print("\n  Por status da fatura:")
print(df.groupby("status_fatura")["consumo_energia_kwh"]
        .agg(media="mean", mediana="median")
        .round(1).to_string())

print("\n  Por competência:")
print(df.groupby("competencia")["consumo_energia_kwh"]
        .agg(media="mean", total="sum")
        .round(1).to_string())

# ══════════════════════════════════════════════════════════
# 3. FATURAMENTO POR TIPO DE CLIENTE
# ══════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  3. FATURAMENTO POR TIPO DE CLIENTE")
print("─" * 60)

fat_tipo = df.groupby("tipo_cliente")["valor_fatura"].agg(
    qtd="count", total="sum", media="mean", mediana="median"
).round(2)
fat_tipo["pct_receita"] = (fat_tipo["total"] / fat_tipo["total"].sum() * 100).round(1)
print(f"\n{fat_tipo.to_string()}")

print("\n  Inadimplência por tipo:")
for tp in df["tipo_cliente"].unique():
    sub  = df[df["tipo_cliente"] == tp]
    atr  = (sub["status_fatura"] == "atrasada").sum()
    taxa = atr / len(sub) * 100
    print(f"    {tp}: {atr}/{len(sub)} atrasadas → {taxa:.1f}%")

# ══════════════════════════════════════════════════════════
# 4. ANÁLISE DE VENCIMENTOS
# ══════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  4. ANÁLISE DE VENCIMENTOS")
print("─" * 60)

dias_top = df["dia_vencimento"].value_counts().sort_values(ascending=False).head(8)
print(f"\n  Top 8 dias de vencimento:\n{dias_top.to_string()}")

print("\n  Taxa de atraso por dia de vencimento (dias com ≥ 5 faturas):")
resumo_dia = (
    df.groupby("dia_vencimento")
      .apply(lambda x: pd.Series({
          "qtd":       len(x),
          "atrasadas": (x["status_fatura"] == "atrasada").sum(),
          "tx_atraso": (x["status_fatura"] == "atrasada").mean() * 100
      }))
      .query("qtd >= 5")
      .sort_values("tx_atraso", ascending=False)
      .round(1)
)
print(resumo_dia.to_string())

# ══════════════════════════════════════════════════════════
# 5. ANÁLISE DE ATRASOS (TEMPO)
# ══════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  5. ATRASO POR COMPETÊNCIA (TENDÊNCIA)")
print("─" * 60)

tend = df.groupby("competencia").apply(lambda x: pd.Series({
    "total":   len(x),
    "atrasadas": (x["status_fatura"] == "atrasada").sum(),
    "tx_atraso": round((x["status_fatura"] == "atrasada").mean() * 100, 1),
    "valor_atrasado": x.loc[x["status_fatura"] == "atrasada", "valor_fatura"].sum()
})).reset_index()

print(f"\n{tend.to_string(index=False)}")
print(f"\n  Variação Jun→Ago: +{tend['tx_atraso'].iloc[-1] - tend['tx_atraso'].iloc[0]:.1f} p.p.")

# ══════════════════════════════════════════════════════════
# 6. CLIENTES VIP
# ══════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  6. CLIENTES VIP (TOP 10 POR VALOR)")
print("─" * 60)

vip = (
    df.groupby(["id_cliente", "tipo_cliente"])
      .agg(
          total_faturado=("valor_fatura", "sum"),
          num_faturas=("valor_fatura", "count"),
          consumo_total=("consumo_energia_kwh", "sum"),
          inadimplente=("status_fatura", lambda x: (x == "atrasada").any())
      )
      .sort_values("total_faturado", ascending=False)
      .head(10)
      .reset_index()
)
vip["rank"] = range(1, 11)
vip["id_curto"] = ["VIP_" + str(i).zfill(2) for i in range(1, 11)]
print(
    vip[["rank","id_curto","tipo_cliente","total_faturado",
         "num_faturas","consumo_total","inadimplente"]]
    .to_string(index=False)
)

# ══════════════════════════════════════════════════════════
# 7. FREQUÊNCIA DE ATRASO POR CLIENTE
# ══════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  7. FREQUÊNCIA DE ATRASO POR CLIENTE")
print("─" * 60)

freq = (
    df.groupby("id_cliente")
      .apply(lambda x: pd.Series({
          "total_fat":   len(x),
          "atrasadas":   (x["status_fatura"] == "atrasada").sum(),
          "tx_atraso":   round((x["status_fatura"] == "atrasada").mean() * 100, 1)
      }))
      .reset_index()
)

print(f"\n  Clientes sem nenhum atraso       : {(freq['atrasadas'] == 0).sum()}")
print(f"  Clientes com algum atraso        : {(freq['atrasadas'] > 0).sum()}")
print(f"  Clientes 100% inadimplentes      : {(freq['tx_atraso'] == 100).sum()}")
print(f"  Clientes com > 50% de atraso     : {(freq['tx_atraso'] > 50).sum()}")
print(f"  Máx. atrasos por cliente         : {int(freq['atrasadas'].max())}")

# distribuição
dist_atr = freq["atrasadas"].value_counts().sort_index()
print(f"\n  Distribuição por qtd de atrasos:")
for k, v in dist_atr.items():
    print(f"    {int(k)} atraso(s): {v} clientes")


# ══════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Gerando gráficos...")
print("=" * 60)

reais = FuncFormatter(lambda x, _: f"R${x:,.0f}")
pct   = FuncFormatter(lambda x, _: f"{x:.0f}%")


# ── FIG 1: Visão Geral de Inadimplência ───────────────────
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
fig1.patch.set_facecolor(BG)
fig1.suptitle("1 · VISÃO GERAL DE INADIMPLÊNCIA", fontsize=13,
              fontweight="bold", color=AMARELO, y=1.02)

# 1a - Pizza de status
ax = axes[0]
ax.set_facecolor(SURFACE)
sizes  = [status_counts.get("paga",0),
          status_counts.get("atrasada",0),
          status_counts.get("em aberto",0)]
labels = ["Paga\n79.3%", "Atrasada\n20.2%", "Em Aberto\n0.4%"]
colors = [VERDE, VERMELHO, TEAL]
wedges, texts = ax.pie(sizes, labels=labels, colors=colors,
                       startangle=90, pctdistance=0.8,
                       wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2))
for t in texts:
    t.set_color(BRANCO); t.set_fontsize(9)
ax.set_title("Status das Faturas", fontsize=10, color=AMARELO, pad=10)
ax.text(0, 0, f"697\nfaturas", ha="center", va="center",
        fontsize=10, color=BRANCO, fontweight="bold")

# 1b - Evolução mensal
ax = axes[1]
comp_sorted = sorted(df["competencia"].unique())
tx_mes = [
    (df[df["competencia"] == c]["status_fatura"] == "atrasada").mean() * 100
    for c in comp_sorted
]
meses_label = ["Jun/21", "Jul/21", "Ago/21"]
bars = ax.bar(meses_label, tx_mes, color=[VERDE, AMARELO, VERMELHO],
              width=0.5, edgecolor=BG, linewidth=1.5)
for bar, v in zip(bars, tx_mes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=10,
            color=BRANCO, fontweight="bold")
ax.set_title("Taxa de Atraso por Competência", fontsize=10, color=AMARELO, pad=10)
ax.set_ylabel("% Atrasadas", color=MUTED)
ax.yaxis.set_major_formatter(pct)
ax.set_ylim(0, 30)

# 1c - Valor atrasado por competência
ax = axes[2]
val_atr = [
    df[(df["competencia"] == c) & (df["status_fatura"] == "atrasada")]["valor_fatura"].sum()
    for c in comp_sorted
]
bars = ax.bar(meses_label, val_atr, color=[VERDE, AMARELO, VERMELHO],
              width=0.5, edgecolor=BG, linewidth=1.5)
for bar, v in zip(bars, val_atr):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"R${v:,.0f}", ha="center", va="bottom", fontsize=8.5,
            color=BRANCO, fontweight="bold")
ax.set_title("Valor em Atraso por Competência", fontsize=10, color=AMARELO, pad=10)
ax.set_ylabel("Valor (R$)", color=MUTED)
ax.yaxis.set_major_formatter(reais)

plt.tight_layout()
fig1.savefig("/home/claude/fig1_inadimplencia.png", dpi=150, bbox_inches="tight",
             facecolor=BG)
print("  ✔ fig1_inadimplencia.png")


# ── FIG 2: Consumo de Energia ─────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.patch.set_facecolor(BG)
fig2.suptitle("2 · COMPORTAMENTO DE CONSUMO DE ENERGIA", fontsize=13,
              fontweight="bold", color=AMARELO, y=1.02)

# 2a - Boxplot consumo por tipo
ax = axes[0]
data_pf = df[df["tipo_cliente"] == "PF"]["consumo_energia_kwh"]
data_pj = df[df["tipo_cliente"] == "PJ"]["consumo_energia_kwh"]
bp = ax.boxplot([data_pf, data_pj], labels=["PF", "PJ"],
                patch_artist=True, notch=False,
                medianprops=dict(color=BG, linewidth=2),
                whiskerprops=dict(color=MUTED),
                capprops=dict(color=MUTED),
                flierprops=dict(marker="o", color=MUTED, alpha=0.5, markersize=3))
bp["boxes"][0].set_facecolor(AMARELO)
bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor(TEAL)
bp["boxes"][1].set_alpha(0.7)
ax.set_title("Distribuição de Consumo por Tipo", fontsize=10, color=AMARELO, pad=10)
ax.set_ylabel("kWh", color=MUTED)

# 2b - Média consumo por status
ax = axes[1]
cons_status = df.groupby("status_fatura")["consumo_energia_kwh"].mean().sort_values()
cores_status = {
    "atrasada": VERMELHO, "paga": VERDE, "em aberto": AMARELO
}
colors_list = [cores_status[s] for s in cons_status.index]
bars = ax.barh(cons_status.index, cons_status.values, color=colors_list,
               edgecolor=BG, height=0.5)
for bar, v in zip(bars, cons_status.values):
    ax.text(v + 3, bar.get_y() + bar.get_height()/2,
            f"{v:.0f} kWh", va="center", fontsize=9, color=BRANCO)
ax.set_title("Consumo Médio por Status", fontsize=10, color=AMARELO, pad=10)
ax.set_xlabel("kWh médio", color=MUTED)

# 2c - Histograma do consumo geral
ax = axes[2]
ax.hist(df["consumo_energia_kwh"], bins=30, color=TEAL, alpha=0.7,
        edgecolor=BG, linewidth=0.5)
ax.axvline(df["consumo_energia_kwh"].median(), color=AMARELO,
           linewidth=1.5, linestyle="--", label=f"Mediana: {df['consumo_energia_kwh'].median():.0f}")
ax.axvline(df["consumo_energia_kwh"].mean(), color=VERMELHO,
           linewidth=1.5, linestyle="--", label=f"Média: {df['consumo_energia_kwh'].mean():.0f}")
ax.set_title("Distribuição de Consumo (todos)", fontsize=10, color=AMARELO, pad=10)
ax.set_xlabel("kWh", color=MUTED)
ax.set_ylabel("Frequência", color=MUTED)
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=MUTED)

plt.tight_layout()
fig2.savefig("/home/claude/fig2_consumo.png", dpi=150, bbox_inches="tight",
             facecolor=BG)
print("  ✔ fig2_consumo.png")


# ── FIG 3: Faturamento por Tipo de Cliente ────────────────
fig3, axes = plt.subplots(1, 3, figsize=(16, 5))
fig3.patch.set_facecolor(BG)
fig3.suptitle("3 · FATURAMENTO POR TIPO DE CLIENTE", fontsize=13,
              fontweight="bold", color=AMARELO, y=1.02)

# 3a - Receita total
ax = axes[0]
receita = df.groupby("tipo_cliente")["valor_fatura"].sum()
bars = ax.bar(receita.index, receita.values,
              color=[AMARELO, TEAL], width=0.4, edgecolor=BG, linewidth=1.5)
for bar, v in zip(bars, receita.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f"R${v:,.0f}", ha="center", fontsize=9, color=BRANCO, fontweight="bold")
ax.set_title("Receita Total por Tipo", fontsize=10, color=AMARELO, pad=10)
ax.yaxis.set_major_formatter(reais)

# 3b - Ticket médio
ax = axes[1]
ticket = df.groupby("tipo_cliente")["valor_fatura"].mean()
bars = ax.bar(ticket.index, ticket.values,
              color=[AMARELO, TEAL], width=0.4, edgecolor=BG, linewidth=1.5)
for bar, v in zip(bars, ticket.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"R${v:.0f}", ha="center", fontsize=9, color=BRANCO, fontweight="bold")
ax.set_title("Ticket Médio por Tipo", fontsize=10, color=AMARELO, pad=10)
ax.yaxis.set_major_formatter(reais)

# 3c - Taxa de atraso por tipo
ax = axes[2]
atr_tipo = df.groupby("tipo_cliente").apply(
    lambda x: (x["status_fatura"] == "atrasada").mean() * 100
)
bars = ax.bar(atr_tipo.index, atr_tipo.values,
              color=[AMARELO, TEAL], width=0.4, edgecolor=BG, linewidth=1.5)
for bar, v in zip(bars, atr_tipo.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{v:.1f}%", ha="center", fontsize=9, color=BRANCO, fontweight="bold")
ax.set_title("Taxa de Inadimplência por Tipo", fontsize=10, color=AMARELO, pad=10)
ax.set_ylabel("% Atrasadas", color=MUTED)
ax.yaxis.set_major_formatter(pct)

plt.tight_layout()
fig3.savefig("/home/claude/fig3_tipo_cliente.png", dpi=150, bbox_inches="tight",
             facecolor=BG)
print("  ✔ fig3_tipo_cliente.png")


# ── FIG 4: Análise de Vencimentos ────────────────────────
fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
fig4.patch.set_facecolor(BG)
fig4.suptitle("4 · ANÁLISE DE VENCIMENTOS DAS FATURAS", fontsize=13,
              fontweight="bold", color=AMARELO, y=1.02)

# 4a - Volume por dia de vencimento
ax = axes[0]
vol_dia = df["dia_vencimento"].value_counts().sort_index()
cores = [VERMELHO if v in [22, 27] else AMARELO if v in [11] else TEAL
         for v in vol_dia.index]
ax.bar(vol_dia.index, vol_dia.values, color=cores, width=0.7,
       edgecolor=BG, linewidth=0.8)
ax.set_title("Volume de Faturas por Dia de Vencimento", fontsize=10,
             color=AMARELO, pad=10)
ax.set_xlabel("Dia do Mês", color=MUTED)
ax.set_ylabel("Qtd. Faturas", color=MUTED)
patch_amarelo = mpatches.Patch(color=AMARELO, label="Dia 11 (melhor)")
patch_verm    = mpatches.Patch(color=VERMELHO, label="Dias 22 e 27 (piores)")
patch_teal    = mpatches.Patch(color=TEAL, label="Demais dias")
ax.legend(handles=[patch_amarelo, patch_verm, patch_teal],
          fontsize=8, facecolor=SURFACE, edgecolor=MUTED)

# 4b - Taxa de atraso por dia (dias com ≥ 5 faturas)
ax = axes[1]
resumo_d = resumo_dia.sort_values("dia_vencimento" if "dia_vencimento" in resumo_dia.columns else resumo_dia.index.name or "index")
resumo_d = resumo_dia.reset_index().sort_values("dia_vencimento")
cores2   = [VERMELHO if v >= 22 else AMARELO if v < 18 else TEAL
            for v in resumo_d["dia_vencimento"]]
ax.scatter(resumo_d["dia_vencimento"], resumo_d["tx_atraso"],
           s=resumo_d["qtd"] * 2.5, c=cores2, alpha=0.8, edgecolors=BG, linewidth=1)
ax.axhline(tx_atrasada, color=VERMELHO, linestyle="--", linewidth=1,
           label=f"Média geral: {tx_atrasada:.1f}%")
for _, row in resumo_d.iterrows():
    if row["qtd"] >= 25:
        ax.annotate(f"Dia {int(row['dia_vencimento'])}\n{row['tx_atraso']:.0f}%",
                    (row["dia_vencimento"], row["tx_atraso"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7.5, color=BRANCO)
ax.set_title("Taxa de Atraso por Dia de Vencimento\n(tamanho = volume)", fontsize=10,
             color=AMARELO, pad=10)
ax.set_xlabel("Dia do Mês", color=MUTED)
ax.set_ylabel("% Atrasadas", color=MUTED)
ax.yaxis.set_major_formatter(pct)
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=MUTED)

plt.tight_layout()
fig4.savefig("/home/claude/fig4_vencimentos.png", dpi=150, bbox_inches="tight",
             facecolor=BG)
print("  ✔ fig4_vencimentos.png")


# ── FIG 5: Clientes VIP ───────────────────────────────────
fig5, axes = plt.subplots(1, 2, figsize=(14, 6))
fig5.patch.set_facecolor(BG)
fig5.suptitle("6 · CLIENTES VIP — TOP 10 POR VALOR FATURADO", fontsize=13,
              fontweight="bold", color=AMARELO, y=1.02)

# 5a - Barras horizontais
ax = axes[0]
vip_sorted = vip.sort_values("total_faturado")
cores_vip  = [VERMELHO if r else VERDE for r in vip_sorted["inadimplente"]]
bars = ax.barh(vip_sorted["id_curto"], vip_sorted["total_faturado"],
               color=cores_vip, edgecolor=BG, height=0.6)
for bar, v in zip(bars, vip_sorted["total_faturado"]):
    ax.text(v + 20, bar.get_y() + bar.get_height()/2,
            f"R${v:,.0f}", va="center", fontsize=8, color=BRANCO)
ax.set_title("Valor Total Faturado", fontsize=10, color=AMARELO, pad=10)
ax.xaxis.set_major_formatter(reais)
p_verm = mpatches.Patch(color=VERMELHO, label="Com atraso")
p_verd = mpatches.Patch(color=VERDE, label="Sem atraso")
ax.legend(handles=[p_verm, p_verd], fontsize=8,
          facecolor=SURFACE, edgecolor=MUTED)

# 5b - Consumo vs faturamento (scatter)
ax = axes[1]
vip_all = (
    df.groupby(["id_cliente","tipo_cliente"])
      .agg(total=("valor_fatura","sum"), consumo=("consumo_energia_kwh","sum"))
      .reset_index()
)
pf_d = vip_all[vip_all["tipo_cliente"] == "PF"]
pj_d = vip_all[vip_all["tipo_cliente"] == "PJ"]
ax.scatter(pf_d["consumo"], pf_d["total"], c=AMARELO, alpha=0.5, s=20,
           edgecolors="none", label="PF")
ax.scatter(pj_d["consumo"], pj_d["total"], c=TEAL, alpha=0.7, s=40,
           edgecolors="none", label="PJ")
# destaque top 10
for _, row in vip.iterrows():
    row_full = vip_all[vip_all["id_cliente"] == row["id_cliente"]]
    if not row_full.empty:
        ax.scatter(row_full["consumo"], row_full["total"],
                   c=VERMELHO, s=80, zorder=5, edgecolors=BRANCO, linewidth=0.8)
ax.set_title("Consumo Total vs Valor Faturado\n(vermelho = TOP 10 VIP)", fontsize=10,
             color=AMARELO, pad=10)
ax.set_xlabel("Consumo Total (kWh)", color=MUTED)
ax.set_ylabel("Valor Total (R$)", color=MUTED)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:,.0f}"))
ax.yaxis.set_major_formatter(reais)
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=MUTED)

plt.tight_layout()
fig5.savefig("/home/claude/fig5_vip.png", dpi=150, bbox_inches="tight",
             facecolor=BG)
print("  ✔ fig5_vip.png")


# ── FIG 6: Frequência de Atraso por Cliente ───────────────
fig6, axes = plt.subplots(1, 3, figsize=(16, 5))
fig6.patch.set_facecolor(BG)
fig6.suptitle("7 · FREQUÊNCIA DE ATRASO POR CLIENTE", fontsize=13,
              fontweight="bold", color=AMARELO, y=1.02)

# 6a - Distribuição de clientes por qtd de atrasos
ax = axes[0]
dist = freq["atrasadas"].value_counts().sort_index()
cores_dist = [VERDE if k == 0 else AMARELO if k == 1 else VERMELHO for k in dist.index]
bars = ax.bar([str(int(k)) for k in dist.index], dist.values,
              color=cores_dist, edgecolor=BG, linewidth=1.5, width=0.5)
for bar, v in zip(bars, dist.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(v), ha="center", fontsize=10, color=BRANCO, fontweight="bold")
ax.set_title("Clientes por Nº de Atrasos", fontsize=10, color=AMARELO, pad=10)
ax.set_xlabel("Qtd. de Atrasos", color=MUTED)
ax.set_ylabel("Nº de Clientes", color=MUTED)

# 6b - Histograma taxa de atraso (só quem atrasa)
ax = axes[1]
atraso_clientes = freq[freq["atrasadas"] > 0]["tx_atraso"]
ax.hist(atraso_clientes, bins=10, color=VERMELHO, alpha=0.75,
        edgecolor=BG, linewidth=0.8)
ax.set_title("Taxa de Atraso — Clientes com ≥ 1 Atraso", fontsize=10,
             color=AMARELO, pad=10)
ax.set_xlabel("% de Faturas Atrasadas", color=MUTED)
ax.set_ylabel("Nº de Clientes", color=MUTED)
ax.axvline(atraso_clientes.mean(), color=AMARELO, linestyle="--", linewidth=1.5,
           label=f"Média: {atraso_clientes.mean():.0f}%")
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=MUTED)

# 6c - Pizza: perfil de adimplência
ax = axes[2]
sem_atraso = (freq["atrasadas"] == 0).sum()
c_100      = (freq["tx_atraso"] == 100).sum()
c_parcial  = ((freq["atrasadas"] > 0) & (freq["tx_atraso"] < 100)).sum()
sizes_p = [sem_atraso, c_parcial, c_100]
labels_p = [f"Sem atraso\n{sem_atraso}", f"Atraso parcial\n{c_parcial}", f"100% inadimp.\n{c_100}"]
wedges2, texts2 = ax.pie(sizes_p, labels=labels_p,
                          colors=[VERDE, AMARELO, VERMELHO],
                          startangle=90,
                          wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2))
for t in texts2:
    t.set_color(BRANCO); t.set_fontsize(9)
ax.set_title("Perfil de Adimplência dos Clientes", fontsize=10, color=AMARELO, pad=10)
ax.text(0, 0, f"{df['id_cliente'].nunique()}\nclientes", ha="center", va="center",
        fontsize=9, color=BRANCO, fontweight="bold")

plt.tight_layout()
fig6.savefig("/home/claude/fig6_frequencia_atraso.png", dpi=150, bbox_inches="tight",
             facecolor=BG)
print("  ✔ fig6_frequencia_atraso.png")


# ── FIG 7: Heatmap correlação e resumo final ──────────────
fig7, axes = plt.subplots(1, 2, figsize=(14, 5))
fig7.patch.set_facecolor(BG)
fig7.suptitle("8 · CORRELAÇÕES E PAINEL DE RISCO", fontsize=13,
              fontweight="bold", color=AMARELO, y=1.02)

# 7a - Heatmap
ax = axes[0]
df_num = df[["valor_fatura","consumo_energia_kwh","dia_vencimento"]].copy()
df_num["inadimplente"] = (df["status_fatura"] == "atrasada").astype(int)
df_num["is_pj"]        = (df["tipo_cliente"] == "PJ").astype(int)
corr = df_num.corr()
sns.heatmap(corr, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
            linewidths=0.5, linecolor=BG,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9},
            xticklabels=["Valor","Consumo","Dia Venc.","Inadimp.","É PJ"],
            yticklabels=["Valor","Consumo","Dia Venc.","Inadimp.","É PJ"])
ax.set_title("Matriz de Correlação", fontsize=10, color=AMARELO, pad=10)
ax.tick_params(colors=BRANCO)

# 7b - Resumo de KPIs como tabela visual
ax = axes[1]
ax.axis("off")
kpis = [
    ["Métrica",                    "Valor",         "Status"],
    ["Total de faturas",           "697",           "—"],
    ["Clientes únicos",            "466",           "—"],
    ["Taxa de inadimplência",      "20.2%",         "⚠ ALTO"],
    ["Valor em atraso",            "R$ 23.715",     "⚠ ALTO"],
    ["Clientes 100% inadimp.",     "97 (20.8%)",    "⚠ ALTO"],
    ["Ticket médio PJ vs PF",      "3.4× maior",    "✔ OPO."],
    ["Melhor dia de vencimento",   "Dia 11 (16.8%)", "✔ BOM"],
    ["Pior dia de vencimento",     "Dia 27 (24.3%)", "⚠ RUIM"],
    ["Tendência inadimplência",    "+9 p.p. em 3m", "⚠ PIORA"],
    ["Clientes sem nenhum atraso", "369 (79.2%)",   "✔ BOM"],
]
tbl = ax.table(cellText=kpis[1:], colLabels=kpis[0],
               loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.3, 1.55)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(MUTED)
    if r == 0:
        cell.set_facecolor(AMARELO)
        cell.set_text_props(color=BG, fontweight="bold")
    elif "⚠" in cell.get_text().get_text():
        cell.set_facecolor("#3d1a1a")
        cell.set_text_props(color=VERMELHO)
    elif "✔" in cell.get_text().get_text():
        cell.set_facecolor("#0f2d1f")
        cell.set_text_props(color=VERDE)
    else:
        cell.set_facecolor(SURFACE)
        cell.set_text_props(color=BRANCO)
ax.set_title("Painel de KPIs — Resumo Executivo", fontsize=10,
             color=AMARELO, pad=10, y=0.98)

plt.tight_layout()
fig7.savefig("/home/claude/fig7_correlacao_kpis.png", dpi=150, bbox_inches="tight",
             facecolor=BG)
print("  ✔ fig7_correlacao_kpis.png")


print("\n" + "=" * 60)
print("  Análise concluída! 7 gráficos gerados.")
print("=" * 60)
