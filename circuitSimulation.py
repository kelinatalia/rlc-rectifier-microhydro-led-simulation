import numpy as np
import matplotlib.pyplot as plt
import os

outDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(outDir, exist_ok=True)

# ac source parameters
vPeak = 5.0          
fSource = 20.0         
omega = 2 * np.pi * fSource
rSrc = 0.1

# bridge rectifier parameters
vDiode = 0.7           
nDiode = 2             

# filter components selection
lFilter = 400e-3        
cFilter = 22e-6         
rLInt = 0.1           
rCInt = 0.1           

# led load parameters
vLed = 2.0          
iLedMax = 20e-3        
nLeds = 3            
rLimit = 100.0        
rRInt = 0.1          

# simulation time parameters
tPeriod = 1.0 / fSource            
tEnd = 15 * tPeriod             
dT = tPeriod / 2000           
nSteps = int(tEnd / dT)
tArr = np.linspace(0, tEnd, nSteps)

def vSource(t):
    # menghitung tegangan ac dari generator
    return vPeak * np.sin(omega * t)

def vRectified(t):
    # menghitung tegangan setelah melewati full wave rectifier
    v = np.abs(vSource(t)) - nDiode * vDiode
    return max(v, 0.0)

def iLoad(vOut, rLimitVal=rLimit):
    # menghitung total arus yang masuk ke 3 cabang led
    if vOut > vLed:
        return nLeds * (vOut - vLed) / (rLimitVal + rRInt)
    return 0.0

def rk4Step(f, t, y, h):
    # solver numerik menggunakan metode runge kutta orde 4
    k1 = f(t,       y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h,   y + h   * k3)
    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulateNoFilter(tArrVal, rLimitVal=rLimit):
    # simulasi rangkaian tanpa filter menggunakan pendekatan aljabar
    n = len(tArrVal)
    vOut = np.zeros(n)
    iLed = np.zeros(n)
    rBranch = rLimitVal + rRInt
    for k in range(n):
        vr = vRectified(tArrVal[k])
        if vr > vLed:
            iTotal = (vr - vLed) / (rSrc + rBranch / nLeds)
            vo = vr - rSrc * iTotal
            vOut[k] = vo
            iLed[k] = iTotal / nLeds
    return vOut, iLed

def simulateCFilter(tArrVal, cVal=cFilter, rLimitVal=rLimit):
    # simulasi filter kapasitor menggunakan rk4
    n = len(tArrVal)
    vC = np.zeros(n)
    iLedArr = np.zeros(n)
    
    def dydt(t, y):
        vc = y[0]
        vr = vRectified(t)
        il = iLoad(vc, rLimitVal)
        if vr > vc:
            iCharge = (vr - vc) / (rSrc + rCInt)
            dvc = (iCharge - il) / cVal
        else:
            dvc = -il / cVal
        return np.array([dvc])
        
    y = np.array([0.0])
    for k in range(n):
        vC[k] = y[0]
        iLedArr[k] = iLoad(y[0], rLimitVal) / nLeds
        if k < n - 1:
            h = tArrVal[k+1] - tArrVal[k]
            y = rk4Step(dydt, tArrVal[k], y, h)
            y[0] = max(y[0], 0.0)
    return vC, iLedArr

def simulateLcFilter(tArrVal, lVal=lFilter, cVal=cFilter, rLimitVal=rLimit):
    # simulasi filter rlc lengkap menggunakan rk4
    n = len(tArrVal)
    ilArr = np.zeros(n)
    vcArr = np.zeros(n)
    iLedArr = np.zeros(n)
    rSer = rSrc + rLInt   
    
    def dydt(t, y):
        iL, vC = y[0], y[1]
        vr = vRectified(t)
        ilOut = iLoad(vC, rLimitVal)
        diL = (vr - rSer * iL - vC) / lVal
        if iL <= 0 and diL < 0:
            diL = 0.0
            iL  = 0.0
        dvC = (max(iL, 0.0) - ilOut) / cVal
        return np.array([diL, dvC])
        
    y = np.array([0.0, 0.0])
    for k in range(n):
        ilArr[k] = y[0]
        vcArr[k] = y[1]
        iLedArr[k] = iLoad(y[1], rLimitVal) / nLeds
        if k < n - 1:
            h = tArrVal[k+1] - tArrVal[k]
            y = rk4Step(dydt, tArrVal[k], y, h)
            y[0] = max(y[0], 0.0)
            y[1] = max(y[1], 0.0)
    return vcArr, iLedArr, ilArr

def steadyStateMetrics(tArrVal, vOut, iLed, label=""):
    # menghitung parameter performa rangkaian pada kondisi steady state
    mask = tArrVal >= (tEnd - 3 * tPeriod)
    vSs = vOut[mask]
    iSs = iLed[mask]
    vMean = np.mean(vSs)
    vMax = np.max(vSs)
    vMin = np.min(vSs)
    vRipple = vMax - vMin
    ripPct = (vRipple / vMean * 100) if vMean > 0 else 0
    iMean = np.mean(iSs) * 1e3
    iMax = np.max(iSs) * 1e3
    iMin = np.min(iSs) * 1e3
    safe = "✓" if iMax <= iLedMax * 1e3 else "✗ over 20ma!"
    metrics = dict(vMean=vMean, vMax=vMax, vMin=vMin,
                   vRipple=vRipple, ripPct=ripPct,
                   iMean=iMean, iMax=iMax, iMin=iMin, safe=safe)
    if label:
        print(f"\n{'─'*50}\n  {label}\n{'─'*50}")
        print(f"  v_dc (mean)  = {vMean:.3f} v")
        print(f"  v_dc (max)   = {vMax:.3f} v")
        print(f"  v_dc (min)   = {vMin:.3f} v")
        print(f"  ripple       = {vRipple:.3f} v  ({ripPct:.1f}%)")
        print(f"  i_led (mean) = {iMean:.2f} ma")
        print(f"  i_led (max)  = {iMax:.2f} ma")
        print(f"  safe (<=20ma) = {safe}")
    return metrics

def main():
    print("=" * 60 + "\n  aol computational physics — rlc rectifier simulation\n" + "=" * 60)
    
    # menjalankan simulasi untuk ketiga konfigurasi sirkuit
    vNf, iNf = simulateNoFilter(tArr)
    vCf, iCf = simulateCFilter(tArr)
    vLc, iLc, ilLc = simulateLcFilter(tArr)
    
    mNf = steadyStateMetrics(tArr, vNf, iNf, "config a – no filter")
    mCf = steadyStateMetrics(tArr, vCf, iCf, "config b – c-only (22 uf)")
    mLc = steadyStateMetrics(tArr, vLc, iLc, "config c – lc (400 mh + 22 uf)")
    
    # analisis variasi nilai r_limit untuk mencari komponen terbaik
    rCandidates = [36, 100, 130, 220, 510]
    print(f"\n{'='*60}\n  r_limit selection analysis\n{'='*60}")
    for r in rCandidates:
        vR, iR2, _ = simulateLcFilter(tArr, rLimitVal=r)
        m2 = steadyStateMetrics(tArr, vR, iR2)
        flag = "✓" if m2['iMax'] <= 20.0 else "✗"
        print(f"  r = {r} ohm | i_max = {m2['iMax']:.2f} ma | safe? = {flag}")
        
    # plotting hasil simulasi sirkuit
    tMs = tArr * 1e3     
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
    
    # membuat grafik overview performa filter
    fig1, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # grafik input ac source
    vSrc = np.array([vSource(t) for t in tArr])
    axes[0,0].plot(tMs, vSrc, 'royalblue', lw=0.8)
    axes[0,0].set_title('(a) ac source voltage')
    axes[0,0].grid(True, alpha=0.3)
    
    # grafik setelah bridge rectifier
    vRec = np.array([vRectified(t) for t in tArr])
    axes[0,1].plot(tMs, vRec, 'orangered', lw=0.8)
    axes[0,1].set_title('(b) full-wave rectified voltage')
    axes[0,1].grid(True, alpha=0.3)
    
    # perbandingan tegangan luaran filter
    axes[1,0].plot(tMs, vNf, 'gray', lw=0.6, alpha=0.7, label='no filter')
    axes[1,0].plot(tMs, vCf, 'orange', lw=0.8, label='c-only')
    axes[1,0].plot(tMs, vLc, 'green', lw=1.0, label='lc filter')
    axes[1,0].axhline(vLed, color='red', ls='--', lw=0.6)
    axes[1,0].set_title('(c) output voltage comparison')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # perbandingan arus pada led
    axes[1,1].plot(tMs, iNf*1e3, 'gray', lw=0.6, alpha=0.7, label='no filter')
    axes[1,1].plot(tMs, iCf*1e3, 'orange', lw=0.8, label='c-only')
    axes[1,1].plot(tMs, iLc*1e3, 'green', lw=1.0, label='lc filter')
    axes[1,1].axhline(20, color='red', ls='--', lw=0.8)
    axes[1,1].set_title('(d) led current comparison')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # grafik arus melewati induktor
    axes[2,0].plot(tMs, ilLc*1e3, 'purple', lw=0.8)
    axes[2,0].set_title('(e) inductor current')
    axes[2,0].grid(True, alpha=0.3)
    
    # grafik zoom steady state filter lc
    zoomStart = (tEnd - 3 * tPeriod) * 1e3
    zm = tMs >= zoomStart
    axes[2,1].plot(tMs[zm], vLc[zm], 'green', lw=1.2)
    axes[2,1].axhline(mLc['vMean'], color='green', ls=':')
    axes[2,1].set_title('(f) steady-state lc output zoom')
    axes[2,1].grid(True, alpha=0.3)
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(outDir, 'fig1_overview.png'), dpi=150)
    
    # membuat grafik perbandingan nilai resistor pembatas
    fig2, (axR1, axR2) = plt.subplots(1, 2, figsize=(13, 5))
    colorsR = {36:'red', 100:'green', 130:'blue', 220:'darkorange', 510:'purple'}
    
    for r in rCandidates:
        vR, iR, _ = simulateLcFilter(tArr, rLimitVal=r)
        zm = tMs >= zoomStart
        ls = '-' if r == 100 else '--'
        lw = 1.4 if r == 100 else 0.8
        axR1.plot(tMs[zm], vR[zm], color=colorsR[r], ls=ls, lw=lw, label=f'r={r} ohm')
        axR2.plot(tMs[zm], iR[zm]*1e3, color=colorsR[r], ls=ls, lw=lw, label=f'r={r} ohm')
        
    axR1.set_title('output voltage steady-state')
    axR1.legend()
    axR1.grid(True, alpha=0.3)
    
    axR2.axhline(20, color='red', ls=':')
    axR2.set_title('led current steady-state')
    axR2.legend()
    axR2.grid(True, alpha=0.3)
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(outDir, 'fig2_resistor.png'), dpi=150)

    # figure 3
    fig3, ax3 = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax3.axis('off')
    topologyText = (
        "CIRCUIT TOPOLOGY — Recommended Design\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "                    ┌──── L (400mH) ────┐\n"
        "    AC Source        │                    │         ┌─ R₁(100Ω) ─ LED₁ ─┐\n"
        "   (5V, 20Hz)  ──► Bridge  ──►          ├── a ──┤─ R₂(100Ω) ─ LED₂ ─├── b\n"
        "    R_int=0.1Ω      Rectifier           │         └─ R₃(100Ω) ─ LED₃ ─┘\n"
        "                    (4 diodes)     C (22μF)                              │\n"
        "                                        │                                │\n"
        "                    GND ────────────────┴────────────────────────────────┘\n\n"
        "  Internal resistance per component: 0.1 ohm\n"
        "  LED specs: V_f = 2V, I_max = 20 mA, no internal resistance"
    )
    ax3.text(0.05, 0.5, topologyText, transform=ax3.transAxes,
             fontfamily='monospace', fontsize=9, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig3.savefig(os.path.join(outDir, 'fig3_topology.png'), dpi=150)

    # figure 4
    fig4, ax4 = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax4.axis('off')
    ax4.set_title('Steady-State Performance Comparison', fontsize=11, fontweight='bold')
    tableData = [
        ['No Filter', f"{mNf['vMean']:.3f}", f"{mNf['vRipple']:.3f}", f"{mNf['ripPct']:.1f}", f"{mNf['iMean']:.2f}", f"{mNf['iMax']:.2f}", mNf['safe']],
        ['C-only (22μF)', f"{mCf['vMean']:.3f}", f"{mCf['vRipple']:.3f}", f"{mCf['ripPct']:.1f}", f"{mCf['iMean']:.2f}", f"{mCf['iMax']:.2f}", mCf['safe']],
        ['LC (400mH+22μF) ★', f"{mLc['vMean']:.3f}", f"{mLc['vRipple']:.3f}", f"{mLc['ripPct']:.1f}", f"{mLc['iMean']:.2f}", f"{mLc['iMax']:.2f}", mLc['safe']],
    ]
    colLabels = ['Configuration', 'V_dc [V]', 'Ripple [V]', 'Ripple [%]', 'I_LED avg [mA]', 'I_LED max [mA]', 'Safe']
    tbl = ax4.table(cellText=tableData, colLabels=colLabels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)
    for j in range(len(colLabels)):
        tbl[3, j].set_facecolor('#d4edda')   
        tbl[0, j].set_facecolor('#e8e8e8')   
        
    fig4.savefig(os.path.join(outDir, 'fig4_comparisonTable.png'), dpi=150)
    plt.show()

if __name__ == '__main__':
    main()