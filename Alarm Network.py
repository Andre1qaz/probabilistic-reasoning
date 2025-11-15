from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def main():
    print("="*60)
    print("PRAKTIKUM BAYESIAN NETWORK - ALARM NETWORK")
    print("Mata Kuliah: Artificial Intelligence")
    print("Institut Teknologi Del")
    print("="*60)

    # 1. Mendefinisikan Struktur Jaringan
    print("\n[1] Membangun Struktur Jaringan...")
    model_alarm = BayesianNetwork([
        ('Burglary', 'Alarm'),
        ('Earthquake', 'Alarm'),
        ('Alarm', 'JohnCalls'),
        ('Alarm', 'MaryCalls')
    ])
    print("✓ Struktur Jaringan Berhasil Dibuat.")
    print("  Nodes:", model_alarm.nodes())
    print("  Edges:", model_alarm.edges())

    # 2. Mendefinisikan Conditional Probability Distributions (CPDs)
    print("\n[2] Mendefinisikan CPDs...")

    # CPD untuk Burglary (B)
    # P(+b) = 0.001
    cpd_b = TabularCPD(variable='Burglary', variable_card=2,
                       values=[[0.999], [0.001]])
    print("  ✓ CPD Burglary: P(B=1) = 0.001")

    # CPD untuk Earthquake (E)
    # P(+e) = 0.002
    cpd_e = TabularCPD(variable='Earthquake', variable_card=2,
                       values=[[0.998], [0.002]])
    print("  ✓ CPD Earthquake: P(E=1) = 0.002")

    # CPD untuk Alarm (A) | Burglary (B), Earthquake (E)
    cpd_a = TabularCPD(variable='Alarm', variable_card=2,
                       values=[[0.999, 0.71, 0.06, 0.05],  # P(A=0 | B, E)
                               [0.001, 0.29, 0.94, 0.95]], # P(A=1 | B, E)
                       evidence=['Burglary', 'Earthquake'],
                       evidence_card=[2, 2])
    print("  ✓ CPD Alarm | Burglary, Earthquake")

    # CPD untuk JohnCalls (J) | Alarm (A)
    cpd_j = TabularCPD(variable='JohnCalls', variable_card=2,
                       values=[[0.95, 0.10],  # P(J=0 | A)
                               [0.05, 0.90]], # P(J=1 | A)
                       evidence=['Alarm'],
                       evidence_card=[2])
    print("  ✓ CPD JohnCalls | Alarm")

    # CPD untuk MaryCalls (M) | Alarm (A)
    cpd_m = TabularCPD(variable='MaryCalls', variable_card=2,
                       values=[[0.99, 0.30],  # P(M=0 | A)
                               [0.01, 0.70]], # P(M=1 | A)
                       evidence=['Alarm'],
                       evidence_card=[2])
    print("  ✓ CPD MaryCalls | Alarm")

    # 3. Menambahkan CPD ke Model
    print("\n[3] Menambahkan CPDs ke Model...")
    model_alarm.add_cpds(cpd_b, cpd_e, cpd_a, cpd_j, cpd_m)
    print("✓ Semua CPDs berhasil ditambahkan")

    # 4. Memverifikasi Model
    print("\n[4] Memverifikasi Model...")
    if model_alarm.check_model():
        print("✓ Model Valid! Semua probabilitas konsisten.")
    else:
        print("✗ Model Tidak Valid!")
        return

    # 5. Melakukan Inferensi dengan Variable Elimination
    print("\n" + "="*60)
    print("INFERENSI PROBABILISTIK")
    print("="*60)

    infer = VariableElimination(model_alarm)

    # Query 1: P(Alarm=1)
    print("\n[Query 1] Probabilitas Alarm berbunyi (tanpa evidence)")
    print("-" * 60)
    q1 = infer.query(variables=['Alarm'])
    print(q1)
    print(f"\nInterpretasi: Alarm sangat jarang berbunyi ({q1.values[1]*100:.2f}%)")
    print("karena burglary dan earthquake adalah kejadian langka.")

    # Query 2: P(Burglary | JohnCalls=1)
    print("\n[Query 2] Probabilitas Burglary jika John menelepon")
    print("-" * 60)
    q2 = infer.query(variables=['Burglary'], evidence={'JohnCalls': 1})
    print(q2)
    print(f"\nInterpretasi: Keyakinan burglary naik dari 0.1% ke {q2.values[1]*100:.2f}%")
    print("ketika John menelepon (peningkatan ~14.5x).")

    # Query 3: P(Burglary | JohnCalls=1, MaryCalls=1)
    print("\n[Query 3] Probabilitas Burglary jika John & Mary menelepon")
    print("-" * 60)
    q3 = infer.query(variables=['Burglary'], 
                     evidence={'JohnCalls': 1, 'MaryCalls': 1})
    print(q3)
    print(f"\nInterpretasi: Dengan 2 evidence, keyakinan melonjak ke {q3.values[1]*100:.2f}%!")
    print("Ini menunjukkan power of multiple evidence dalam Bayesian reasoning.")

    # Query 4: P(Earthquake | JohnCalls=1, MaryCalls=1)
    print("\n[Query 4] Probabilitas Earthquake jika John & Mary menelepon")
    print("-" * 60)
    q4 = infer.query(variables=['Earthquake'], 
                     evidence={'JohnCalls': 1, 'MaryCalls': 1})
    print(q4)
    print(f"\nInterpretasi: Earthquake juga naik ke {q4.values[1]*100:.2f}%")
    print("Ini adalah contoh 'Common Effect' - kedua causes (B & E) compete")
    print("untuk menjelaskan effect (Alarm).")

    # Summary
    print("\n" + "="*60)
    print("RINGKASAN HASIL")
    print("="*60)
    print(f"Prior P(Burglary=1)                    : 0.10%")
    print(f"P(Burglary=1 | JohnCalls=1)            : {q2.values[1]*100:.2f}%")
    print(f"P(Burglary=1 | John+Mary)              : {q3.values[1]*100:.2f}%")
    print(f"P(Earthquake=1 | John+Mary)            : {q4.values[1]*100:.2f}%")
    print("\nKesimpulan: Multiple evidence secara dramatis meningkatkan")
    print("keyakinan kita tentang causes yang mendasari observations.")

    print("\n" + "="*60)
    print("PRAKTIKUM SELESAI")
    print("="*60)
    
    return infer

if __name__ == "__main__":
    inference_engine = main()
    
    # Anda bisa menambahkan queries tambahan di sini
    # Contoh:
    # result = inference_engine.query(variables=['Alarm'], 
    #                                 evidence={'Burglary': 1})
    # print(result)