"""
LATIHAN MANDIRI 2: SPRINKLER NETWORK (WetGrass)
Mata Kuliah: Artificial Intelligence (10S3001)
Institut Teknologi Del

Tujuan: Membangun Bayesian Network baru untuk studi kasus
"WetGrass" dan memahami probabilistic reasoning dalam konteks
yang berbeda.

Nama: [Isi Nama Anda]
NIM: [Isi NIM Anda]
"""
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork 
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def build_sprinkler_network():
    """
    Membangun Sprinkler Network
    
    Struktur:
        Cloudy
        /    \
       v      v
    Sprinkler  Rain
        \      /
         v    v
        WetGrass
    
    Variabel:
    - Cloudy (C): Mendung (T/F)
    - Sprinkler (S): Sprinkler menyala (T/F)
    - Rain (R): Hujan (T/F)
    - WetGrass (W): Rumput basah (T/F)
    """
    
    print("="*70)
    print("MEMBANGUN SPRINKLER NETWORK")
    print("="*70)
    
    # 1. Struktur Network
    print("\n[1] Mendefinisikan Struktur...")
    model = BayesianNetwork([
        ('Cloudy', 'Sprinkler'),
        ('Cloudy', 'Rain'),
        ('Sprinkler', 'WetGrass'),
        ('Rain', 'WetGrass')
    ])
    print("✓ Struktur berhasil dibuat")
    print("  Nodes:", list(model.nodes()))
    print("  Edges:", list(model.edges()))
    
    # 2. CPD untuk Cloudy (root node)
    print("\n[2] Mendefinisikan CPDs...")
    # P(Cloudy=1) = 0.5 (50% chance mendung)
    cpd_c = TabularCPD('Cloudy', 2, 
                       [[0.5],   # P(C=0)
                        [0.5]])  # P(C=1)
    print("  ✓ CPD Cloudy: P(C=1) = 0.5")
    
    # 3. CPD untuk Sprinkler | Cloudy
    # Logika: Sprinkler jarang menyala saat mendung
    # P(S=1|C=0) = 0.5, P(S=1|C=1) = 0.1
    cpd_s = TabularCPD('Sprinkler', 2,
                       [[0.5, 0.9],   # P(S=0|C=0), P(S=0|C=1)
                        [0.5, 0.1]],  # P(S=1|C=0), P(S=1|C=1)
                       evidence=['Cloudy'],
                       evidence_card=[2])
    print("  ✓ CPD Sprinkler | Cloudy")
    print("    P(S=1|C=0) = 0.5, P(S=1|C=1) = 0.1")
    
    # 4. CPD untuk Rain | Cloudy
    # Logika: Hujan lebih sering saat mendung
    # P(R=1|C=0) = 0.2, P(R=1|C=1) = 0.8
    cpd_r = TabularCPD('Rain', 2,
                       [[0.8, 0.2],   # P(R=0|C=0), P(R=0|C=1)
                        [0.2, 0.8]],  # P(R=1|C=0), P(R=1|C=1)
                       evidence=['Cloudy'],
                       evidence_card=[2])
    print("  ✓ CPD Rain | Cloudy")
    print("    P(R=1|C=0) = 0.2, P(R=1|C=1) = 0.8")
    
    # 5. CPD untuk WetGrass | Sprinkler, Rain
    # Logika: Rumput basah jika sprinkler menyala ATAU hujan (atau keduanya)
    # Tabel: [S=0,R=0], [S=0,R=1], [S=1,R=0], [S=1,R=1]
    cpd_w = TabularCPD('WetGrass', 2,
                       [[1.0, 0.1, 0.1, 0.01],    # P(W=0|S,R)
                        [0.0, 0.9, 0.9, 0.99]],   # P(W=1|S,R)
                       evidence=['Sprinkler', 'Rain'],
                       evidence_card=[2, 2])
    print("  ✓ CPD WetGrass | Sprinkler, Rain")
    print("    P(W=1|S=0,R=0) = 0.0  (tidak ada yang bikin basah)")
    print("    P(W=1|S=0,R=1) = 0.9  (hujan saja)")
    print("    P(W=1|S=1,R=0) = 0.9  (sprinkler saja)")
    print("    P(W=1|S=1,R=1) = 0.99 (hujan DAN sprinkler)")
    
    # 6. Tambahkan CPDs ke model
    print("\n[3] Menambahkan CPDs ke model...")
    model.add_cpds(cpd_c, cpd_s, cpd_r, cpd_w)
    print("✓ Semua CPDs berhasil ditambahkan")
    
    # 7. Validasi
    print("\n[4] Validasi model...")
    if model.check_model():
        print("✓ Model Valid!")
    else:
        print("✗ Model Tidak Valid!")
        raise ValueError("Model validation failed")
    
    return model

def perform_inference(model):
    """Melakukan berbagai inferensi pada Sprinkler Network"""
    
    print("\n" + "="*70)
    print("INFERENSI PROBABILISTIK")
    print("="*70)
    
    infer = VariableElimination(model)
    
    # Query 1: Marginal probabilities
    print("\n[Query 1] Probabilitas Marginal (tanpa evidence)")
    print("-" * 70)
    for var in ['Cloudy', 'Sprinkler', 'Rain', 'WetGrass']:
        result = infer.query(variables=[var])
        print(f"P({var}=1) = {result.values[1]:.4f} ({result.values[1]*100:.2f}%)")
    
    # Query 2: P(Rain=1 | WetGrass=1) - TUGAS UTAMA
    print("\n[Query 2] P(Rain=1 | WetGrass=1)")
    print("-" * 70)
    print("PERTANYAAN: Jika rumput basah, berapa probabilitas hujan?")
    q2 = infer.query(variables=['Rain'], evidence={'WetGrass': 1})
    print(q2)
    prob_rain_wet = q2.values[1]
    print(f"\n✓ Probabilitas hujan jika rumput basah: {prob_rain_wet*100:.2f}%")
    print("  Interpretasi: Rumput basah meningkatkan keyakinan hujan dari")
    print(f"  prior ~50% menjadi {prob_rain_wet*100:.1f}%.")
    
    # Query 3: Explaining Away - Sprinkler explains away Rain
    print("\n[Query 3] P(Rain=1 | WetGrass=1, Sprinkler=1)")
    print("-" * 70)
    print("PERTANYAAN: Jika rumput basah DAN sprinkler menyala,")
    print("             berapa probabilitas hujan?")
    q3 = infer.query(variables=['Rain'], 
                     evidence={'WetGrass': 1, 'Sprinkler': 1})
    print(q3)
    prob_rain_wet_sprinkler = q3.values[1]
    print(f"\n✓ Probabilitas hujan: {prob_rain_wet_sprinkler*100:.2f}%")
    print(f"  Perubahan: TURUN dari {prob_rain_wet*100:.1f}% ke {prob_rain_wet_sprinkler*100:.1f}%")
    print("  Interpretasi: Sprinkler 'explains away' rumput basah,")
    print("  sehingga hujan menjadi kurang probable.")
    
    # Query 4: Reverse explaining away
    print("\n[Query 4] P(Rain=1 | WetGrass=1, Sprinkler=0)")
    print("-" * 70)
    print("PERTANYAAN: Jika rumput basah DAN sprinkler TIDAK menyala,")
    print("             berapa probabilitas hujan?")
    q4 = infer.query(variables=['Rain'], 
                     evidence={'WetGrass': 1, 'Sprinkler': 0})
    print(q4)
    prob_rain_wet_no_sprinkler = q4.values[1]
    print(f"\n✓ Probabilitas hujan: {prob_rain_wet_no_sprinkler*100:.2f}%")
    print(f"  Perubahan: NAIK dari {prob_rain_wet*100:.1f}% ke {prob_rain_wet_no_sprinkler*100:.1f}%!")
    print("  Interpretasi: Jika sprinkler TIDAK menyala tapi rumput basah,")
    print("  maka hujan HARUS yang menyebabkan rumput basah.")
    
    # Query 5: Diagnostic reasoning - dari effect ke causes
    print("\n[Query 5] P(Sprinkler, Rain | WetGrass=1)")
    print("-" * 70)
    print("PERTANYAAN: Jika rumput basah, probabilitas KEDUA causes?")
    q5 = infer.query(variables=['Sprinkler', 'Rain'], 
                     evidence={'WetGrass': 1})
    print(q5)
    
    # Analysis
    print("\n" + "="*70)
    print("ANALISIS EXPLAINING AWAY")
    print("="*70)
    print(f"""
Ringkasan hasil:
1. P(Rain=1 | WetGrass=1)                    : {prob_rain_wet*100:.1f}%
2. P(Rain=1 | WetGrass=1, Sprinkler=1)       : {prob_rain_wet_sprinkler*100:.1f}%
3. P(Rain=1 | WetGrass=1, Sprinkler=0)       : {prob_rain_wet_no_sprinkler*100:.1f}%

Fenomena Explaining Away:
- Ketika sprinkler MENYALA  → Rain probability TURUN ({prob_rain_wet*100:.1f}% → {prob_rain_wet_sprinkler*100:.1f}%)
- Ketika sprinkler TIDAK    → Rain probability NAIK  ({prob_rain_wet*100:.1f}% → {prob_rain_wet_no_sprinkler*100:.1f}%)

Penjelasan:
- Rain dan Sprinkler adalah competing causes untuk WetGrass
- Evidence untuk satu cause mengurangi probability cause lainnya
- Negative evidence (Sprinkler=0) meningkatkan probability cause lain!
- Ini adalah contoh klasik intercausal reasoning dalam Bayesian Network

Perbandingan dengan Alarm Network:
- Alarm Network: Burglary & Earthquake → Alarm
- Sprinkler Network: Rain & Sprinkler → WetGrass
- Keduanya menunjukkan fenomena explaining away yang sama!
    """)
    
    return infer

def additional_queries(infer):
    """Queries tambahan untuk eksplorasi"""
    
    print("\n" + "="*70)
    print("QUERIES TAMBAHAN (EKSPLORASI)")
    print("="*70)
    
    # Query: Predictive reasoning (dari cause ke effect)
    print("\n[Extra 1] Predictive Reasoning: P(WetGrass=1 | Rain=1)")
    q_pred = infer.query(variables=['WetGrass'], evidence={'Rain': 1})
    print(q_pred)
    print(f"Jika hujan, probabilitas rumput basah: {q_pred.values[1]*100:.1f}%")
    
    # Query: Common cause effect
    print("\n[Extra 2] Common Cause: P(Sprinkler, Rain | Cloudy=1)")
    q_cc = infer.query(variables=['Sprinkler', 'Rain'], 
                       evidence={'Cloudy': 1})
    print(q_cc)
    print("Cloudy adalah 'common cause' untuk Sprinkler dan Rain")
    
    # Query: Independence check
    print("\n[Extra 3] Conditional Independence")
    q_ind1 = infer.query(variables=['Sprinkler'], evidence={'Rain': 1})
    q_ind2 = infer.query(variables=['Sprinkler'])
    print(f"P(Sprinkler=1)              : {q_ind2.values[1]:.4f}")
    print(f"P(Sprinkler=1 | Rain=1)     : {q_ind1.values[1]:.4f}")
    if abs(q_ind1.values[1] - q_ind2.values[1]) < 0.01:
        print("✓ Sprinkler dan Rain relatif independent (tidak di-condition)")
    else:
        print("✗ Sprinkler dan Rain TIDAK independent")
    
    print("\nSekarang condition pada WetGrass:")
    q_dep = infer.query(variables=['Sprinkler'], 
                        evidence={'Rain': 1, 'WetGrass': 1})
    print(f"P(Sprinkler=1 | Rain=1, WetGrass=1): {q_dep.values[1]:.4f}")
    print("✓ Sprinkler dan Rain menjadi DEPENDENT ketika di-condition")
    print("  pada WetGrass (common effect)!")

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║           LATIHAN 2: SPRINKLER NETWORK (WetGrass)                 ║
    ║                   Bayesian Network Tutorial                       ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Build network
    try:
        model = build_sprinkler_network()
    except Exception as e:
        print(f"\n✗ Error saat membangun network: {str(e)}")
        print("\nPastikan pgmpy terinstall dengan benar:")
        print("  pip install pgmpy --upgrade")
        return
    
    # Perform main inference
    try:
        infer = perform_inference(model)
    except Exception as e:
        print(f"\n✗ Error saat inferensi: {str(e)}")
        return
    
    # Additional queries
    try:
        additional_queries(infer)
    except Exception as e:
        print(f"\n✗ Error saat additional queries: {str(e)}")
    
    # Final summary
    print("\n" + "="*70)
    print("KESIMPULAN LATIHAN 2")
    print("="*70)
    print("""
✓ Berhasil membangun Sprinkler Network dengan 4 variabel
✓ Memahami struktur "V-structure" (collider): S → W ← R
✓ Mendemonstrasikan explaining away effect
✓ Membandingkan diagnostic vs predictive reasoning
✓ Memahami conditional independence

Key Insights:
1. Common effect (WetGrass) membuat causes (Rain, Sprinkler) dependent
2. Explaining away bekerja dalam kedua arah (symmetric)
3. Negative evidence (Sprinkler=0) sama powerful dengan positive evidence
4. Bayesian Network naturally handles complex probabilistic dependencies

Aplikasi Real-World:
- Weather prediction systems
- Sensor fusion in robotics
- Causal reasoning in decision making
- Diagnostic systems (medical, technical)
    """)
    
    print("\n" + "="*70)
    print("LATIHAN 2 SELESAI")
    print("="*70)
    print("\nSaran: Coba modify CPD values dan lihat bagaimana hasil berubah!")
    print("Eksplorasi: Tambahkan node baru seperti 'SlipperyPath' atau 'NeedUmbrella'")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram dihentikan oleh user.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        print("\nJika masalah berlanjut, hubungi asisten praktikum.")
        import traceback
        traceback.print_exc()