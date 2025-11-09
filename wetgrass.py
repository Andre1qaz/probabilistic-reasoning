"""
LATIHAN MANDIRI 2: STUDI KASUS SPRINKLER (WETGRASS)
Bayesian Network untuk memodelkan penyebab rumput basah
"""

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

print("="*60)
print("LATIHAN MANDIRI 2 - SPRINKLER/WETGRASS NETWORK")
print("="*60)

# 1. Mendefinisikan Struktur Jaringan
# Variabel: Cloudy (C), Sprinkler (S), Rain (R), WetGrass (W)
# Struktur: C -> S, C -> R, S -> W, R -> W
model_sprinkler = BayesianNetwork([
    ('Cloudy', 'Sprinkler'),
    ('Cloudy', 'Rain'),
    ('Sprinkler', 'WetGrass'),
    ('Rain', 'WetGrass')
])

print("\n✓ Struktur Jaringan Berhasil Dibuat.")
print("  Variabel: Cloudy, Sprinkler, Rain, WetGrass")
print("  Nodes:", model_sprinkler.nodes())
print("  Edges:", model_sprinkler.edges())
print("\nPenjelasan Struktur:")
print("  • Cloudy (berawan) mempengaruhi Sprinkler dan Rain")
print("  • Sprinkler dan Rain keduanya mempengaruhi WetGrass")

# 2. Mendefinisikan CPDs
print("\n" + "-"*60)
print("Mendefinisikan Conditional Probability Distributions...")
print("-"*60)

# CPD untuk Cloudy (C)
# P(Cloudy=1) = 0.5 (asumsi: 50% kemungkinan berawan)
cpd_c = TabularCPD(variable='Cloudy', variable_card=2,
                   values=[[0.5],   # P(C=0) - tidak berawan
                           [0.5]])  # P(C=1) - berawan

print("\nCPD Cloudy:")
print("  P(Cloudy=0) = 0.5")
print("  P(Cloudy=1) = 0.5")

# CPD untuk Sprinkler (S) | Cloudy (C)
# Logika: Jika berawan (C=1), sprinkler jarang menyala
# C=0 (tidak berawan): P(S=1) = 0.5
# C=1 (berawan): P(S=1) = 0.1
cpd_s = TabularCPD(variable='Sprinkler', variable_card=2,
                   values=[[0.5, 0.9],   # P(S=0 | C)
                           [0.5, 0.1]],  # P(S=1 | C)
                   evidence=['Cloudy'],
                   evidence_card=[2])

print("\nCPD Sprinkler | Cloudy:")
print("  P(Sprinkler=1 | Cloudy=0) = 0.5  (tidak berawan → sprinkler sering)")
print("  P(Sprinkler=1 | Cloudy=1) = 0.1  (berawan → sprinkler jarang)")

# CPD untuk Rain (R) | Cloudy (C)
# Logika: Jika berawan (C=1), lebih mungkin hujan
# C=0 (tidak berawan): P(R=1) = 0.2
# C=1 (berawan): P(R=1) = 0.8
cpd_r = TabularCPD(variable='Rain', variable_card=2,
                   values=[[0.8, 0.2],   # P(R=0 | C)
                           [0.2, 0.8]],  # P(R=1 | C)
                   evidence=['Cloudy'],
                   evidence_card=[2])

print("\nCPD Rain | Cloudy:")
print("  P(Rain=1 | Cloudy=0) = 0.2  (tidak berawan → jarang hujan)")
print("  P(Rain=1 | Cloudy=1) = 0.8  (berawan → sering hujan)")

# CPD untuk WetGrass (W) | Sprinkler (S), Rain (R)
# Logika: Rumput basah jika sprinkler menyala ATAU hujan (atau keduanya)
# S=0, R=0: P(W=1) = 0.0   (tidak ada yang membasahi)
# S=0, R=1: P(W=1) = 0.9   (hujan saja)
# S=1, R=0: P(W=1) = 0.9   (sprinkler saja)
# S=1, R=1: P(W=1) = 0.99  (hujan DAN sprinkler)
cpd_w = TabularCPD(variable='WetGrass', variable_card=2,
                   values=[[1.0,  0.1,  0.1,  0.01],  # P(W=0 | S, R)
                           [0.0,  0.9,  0.9,  0.99]],  # P(W=1 | S, R)
                   evidence=['Sprinkler', 'Rain'],
                   evidence_card=[2, 2])

print("\nCPD WetGrass | Sprinkler, Rain:")
print("  P(WetGrass=1 | S=0, R=0) = 0.00  (tidak ada → rumput kering)")
print("  P(WetGrass=1 | S=0, R=1) = 0.90  (hanya hujan)")
print("  P(WetGrass=1 | S=1, R=0) = 0.90  (hanya sprinkler)")
print("  P(WetGrass=1 | S=1, R=1) = 0.99  (keduanya)")

# 3. Menambahkan CPD ke Model
model_sprinkler.add_cpds(cpd_c, cpd_s, cpd_r, cpd_w)

# 4. Memverifikasi Model
if model_sprinkler.check_model():
    print("\n✓ Model Berhasil Dibuat dan Valid.")
else:
    print("\n✗ Struktur Model atau CPD Tidak Valid.")

# 5. Melakukan Inferensi
print("\n" + "="*60)
print("INFERENSI")
print("="*60)

infer = VariableElimination(model_sprinkler)

# TUGAS UTAMA: P(Rain=1 | WetGrass=1)
print("\n[TUGAS] Probabilitas hujan, jika rumput basah:")
print("        P(Rain=1 | WetGrass=1)")
q_main = infer.query(variables=['Rain'], evidence={'WetGrass': 1})
print(q_main)

# Query tambahan untuk analisis
print("\n[Q1] Probabilitas hujan (tanpa bukti):")
print("     P(Rain=1)")
q1 = infer.query(variables=['Rain'])
print(q1)

print("\n[Q2] Probabilitas sprinkler menyala, jika rumput basah:")
print("     P(Sprinkler=1 | WetGrass=1)")
q2 = infer.query(variables=['Sprinkler'], evidence={'WetGrass': 1})
print(q2)

print("\n[Q3] Probabilitas hujan, jika rumput basah DAN sprinkler menyala:")
print("     P(Rain=1 | WetGrass=1, Sprinkler=1)")
q3 = infer.query(variables=['Rain'], evidence={'WetGrass': 1, 'Sprinkler': 1})
print(q3)

print("\n[Q4] Probabilitas hujan, jika rumput basah DAN sprinkler TIDAK menyala:")
print("     P(Rain=1 | WetGrass=1, Sprinkler=0)")
q4 = infer.query(variables=['Rain'], evidence={'WetGrass': 1, 'Sprinkler': 0})
print(q4)

print("\n[Q5] Probabilitas berawan, jika rumput basah:")
print("     P(Cloudy=1 | WetGrass=1)")
q5 = infer.query(variables=['Cloudy'], evidence={'WetGrass': 1})
print(q5)

# Analisis
print("\n" + "="*60)
print("ANALISIS HASIL")
print("="*60)

p_rain = q1.values[1]
p_rain_given_wet = q_main.values[1]
p_sprinkler_given_wet = q2.values[1]
p_rain_given_wet_sprinkler = q3.values[1]
p_rain_given_wet_no_sprinkler = q4.values[1]

print(f"""
HASIL TUGAS UTAMA:
P(Rain=1 | WetGrass=1) = {p_rain_given_wet:.4f}

Artinya: Jika kita tahu rumput basah, probabilitas hujan adalah {p_rain_given_wet*100:.2f}%

ANALISIS MENDALAM:

1. PRIOR vs POSTERIOR:
   • P(Rain=1) = {p_rain:.4f} (sebelum tahu rumput basah)
   • P(Rain=1 | WetGrass=1) = {p_rain_given_wet:.4f} (setelah tahu rumput basah)
   → Probabilitas hujan NAIK ketika kita tahu rumput basah

2. KEDUA PENYEBAB SAMA-SAMA MUNGKIN:
   • P(Rain=1 | WetGrass=1) = {p_rain_given_wet:.4f}
   • P(Sprinkler=1 | WetGrass=1) = {p_sprinkler_given_wet:.4f}
   → Rumput basah bisa karena hujan ATAU sprinkler

3. EXPLAINING AWAY (lagi!):
   • P(Rain=1 | WetGrass=1, Sprinkler=1) = {p_rain_given_wet_sprinkler:.4f}
     → Jika sprinkler menyala, probabilitas hujan TURUN
   
   • P(Rain=1 | WetGrass=1, Sprinkler=0) = {p_rain_given_wet_no_sprinkler:.4f}
     → Jika sprinkler TIDAK menyala, probabilitas hujan NAIK
   
   → Ini adalah "Explaining Away": Mengetahui satu penyebab (sprinkler)
     menjelaskan efek (rumput basah), sehingga penyebab lain (hujan)
     menjadi kurang diperlukan untuk menjelaskan observasi.

KESIMPULAN:
Bayesian Network dapat memodelkan reasoning yang kompleks dan intuitif:
- Rumput basah meningkatkan keyakinan akan hujan
- Tapi jika kita tahu sprinkler menyala, kita jadi kurang yakin hujan
- Model ini menangkap ketidakpastian dengan elegant!
""")

print("="*60)
print("Program selesai!")
print("="*60)