"""
Praktikum: Penalaran Probabilistik dengan Bayesian Network
Topik: Alarm Network
Tanggal: 9 November 2025
"""

# Import library yang diperlukan
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

print("="*60)
print("PRAKTIKUM BAYESIAN NETWORK - ALARM NETWORK")
print("="*60)

# 1. Mendefinisikan Struktur Jaringan
# Variabel: Burglary (B), Earthquake (E), Alarm (A), JohnCalls (J), MaryCalls (M)
# Struktur: (B, A), (E, A), (A, J), (A, M)
model_alarm = BayesianNetwork([
    ('Burglary', 'Alarm'),
    ('Earthquake', 'Alarm'),
    ('Alarm', 'JohnCalls'),
    ('Alarm', 'MaryCalls')
])
print("\n✓ Struktur Jaringan Berhasil Dibuat.")
print("  Nodes:", model_alarm.nodes())
print("  Edges:", model_alarm.edges())

# 2. Mendefinisikan Conditional Probability Distributions (CPDs)
# Kita gunakan 0 untuk False dan 1 untuk True

# CPD untuk Burglary (B)
# P(+b) = 0.001
cpd_b = TabularCPD(variable='Burglary', variable_card=2,
                   values=[[0.999], [0.001]])  # [P(B=0), P(B=1)]

# CPD untuk Earthquake (E)
# P(+e) = 0.002
cpd_e = TabularCPD(variable='Earthquake', variable_card=2,
                   values=[[0.998], [0.002]])  # [P(E=0), P(E=1)]

# CPD untuk Alarm (A) | Burglary (B), Earthquake (E)
# P(A | B, E)
# B=0, E=0: P(A=0)=0.999, P(A=1)=0.001
# B=0, E=1: P(A=0)=0.71, P(A=1)=0.29
# B=1, E=0: P(A=0)=0.06, P(A=1)=0.94
# B=1, E=1: P(A=0)=0.05, P(A=1)=0.95
cpd_a = TabularCPD(variable='Alarm', variable_card=2,
                   values=[[0.999, 0.71, 0.06, 0.05],  # P(A=0 | B, E)
                           [0.001, 0.29, 0.94, 0.95]],  # P(A=1 | B, E)
                   evidence=['Burglary', 'Earthquake'],
                   evidence_card=[2, 2])  # Jumlah state B dan E

# CPD untuk JohnCalls (J) | Alarm (A)
# P(J | A)
# A=0: P(J=0)=0.95, P(J=1)=0.05
# A=1: P(J=0)=0.10, P(J=1)=0.90
cpd_j = TabularCPD(variable='JohnCalls', variable_card=2,
                   values=[[0.95, 0.10],  # P(J=0 | A)
                           [0.05, 0.90]],  # P(J=1 | A)
                   evidence=['Alarm'],
                   evidence_card=[2])  # Jumlah state A

# CPD untuk MaryCalls (M) | Alarm (A)
# P(M | A)
# A=0: P(M=0)=0.99, P(M=1)=0.01
# A=1: P(M=0)=0.30, P(M=1)=0.70
cpd_m = TabularCPD(variable='MaryCalls', variable_card=2,
                   values=[[0.99, 0.30],  # P(M=0 | A)
                           [0.01, 0.70]],  # P(M=1 | A)
                   evidence=['Alarm'],
                   evidence_card=[2])  # Jumlah state A

# 3. Menambahkan CPD ke Model
model_alarm.add_cpds(cpd_b, cpd_e, cpd_a, cpd_j, cpd_m)

# 4. Memverifikasi Model
if model_alarm.check_model():
    print("\n✓ Model Berhasil Dibuat dan Valid.")
else:
    print("\n✗ Struktur Model atau CPD Tidak Valid.")

# 5. Melakukan Inferensi dengan Variable Elimination
print("\n" + "="*60)
print("INFERENSI - PERTANYAAN DARI MODUL")
print("="*60)

infer = VariableElimination(model_alarm)

# Pertanyaan 1: Query Sederhana
print("\n[Q1] Probabilitas Alarm berbunyi (tanpa bukti apapun):")
print("     P(Alarm=1)")
q1 = infer.query(variables=['Alarm'])
print(q1)

# Pertanyaan 2: Query Diagnostik
print("\n[Q2] Probabilitas Perampokan, jika John menelepon:")
print("     P(Burglary=1 | JohnCalls=1)")
q2 = infer.query(variables=['Burglary'], evidence={'JohnCalls': 1})
print(q2)

# Pertanyaan 3: Query Diagnostik Kompleks
print("\n[Q3] Probabilitas Perampokan, jika John & Mary menelepon:")
print("     P(Burglary=1 | JohnCalls=1, MaryCalls=1)")
q3 = infer.query(variables=['Burglary'], evidence={'JohnCalls': 1, 'MaryCalls': 1})
print(q3)

# Pertanyaan 4: Common Cause
print("\n[Q4] Probabilitas Gempa, jika John & Mary menelepon:")
print("     P(Earthquake=1 | JohnCalls=1, MaryCalls=1)")
q4 = infer.query(variables=['Earthquake'], evidence={'JohnCalls': 1, 'MaryCalls': 1})
print(q4)

# LATIHAN MANDIRI 1: Common Effect
print("\n" + "="*60)
print("LATIHAN MANDIRI 1 - COMMON EFFECT")
print("="*60)

print("\n[L1.1] Probabilitas perampokan (tanpa bukti):")
print("       P(Burglary=1)")
l1_1 = infer.query(variables=['Burglary'])
print(l1_1)

print("\n[L1.2] Probabilitas perampokan, jika alarm berbunyi:")
print("       P(Burglary=1 | Alarm=1)")
l1_2 = infer.query(variables=['Burglary'], evidence={'Alarm': 1})
print(l1_2)

print("\n[L1.3] Probabilitas perampokan, jika alarm berbunyi DAN terjadi gempa:")
print("       P(Burglary=1 | Alarm=1, Earthquake=1)")
l1_3 = infer.query(variables=['Burglary'], evidence={'Alarm': 1, 'Earthquake': 1})
print(l1_3)

print("\n" + "="*60)
print("ANALISIS HASIL")
print("="*60)

print("""
ANALISIS PERTANYAAN MODUL:
1. P(Alarm=1) = 0.0025 (sangat kecil)
   → Alarm jarang berbunyi karena perampokan dan gempa jarang terjadi

2. P(Burglary=1 | JohnCalls=1) = 0.0145
   → Keyakinan perampokan naik dari 0.001 ke 0.0145 (naik ~14x)

3. P(Burglary=1 | JohnCalls=1, MaryCalls=1) = 0.2978
   → Dengan dua bukti, keyakinan perampokan meningkat drastis (~298x)
   → Ini menunjukkan kekuatan penalaran probabilistik!

4. P(Earthquake=1 | JohnCalls=1, MaryCalls=1) = 0.1792
   → Gempa juga menjadi penjelasan yang mungkin untuk alarm berbunyi

ANALISIS LATIHAN 1 (Common Effect):
""")

# Ekstrak nilai probabilitas
p_b = l1_1.values[1]
p_b_given_a = l1_2.values[1]
p_b_given_a_e = l1_3.values[1]

print(f"• P(Burglary=1) = {p_b:.4f}")
print(f"• P(Burglary=1 | Alarm=1) = {p_b_given_a:.4f}")
print(f"• P(Burglary=1 | Alarm=1, Earthquake=1) = {p_b_given_a_e:.4f}")

print(f"""
KESIMPULAN:
- Ketika alarm berbunyi, probabilitas perampokan naik drastis
  (dari {p_b:.4f} ke {p_b_given_a:.4f})
  
- NAMUN, ketika kita tahu ada gempa (penyebab lain alarm), probabilitas
  perampokan TURUN menjadi {p_b_given_a_e:.4f}
  
- Ini adalah contoh "Explaining Away": Mengetahui satu penyebab (gempa)
  mengurangi keyakinan kita pada penyebab lain (perampokan), meskipun
  efeknya (alarm) tetap sama.
""")

print("="*60)
print("Program selesai!")
print("="*60)