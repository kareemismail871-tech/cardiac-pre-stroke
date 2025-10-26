import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

# ==============================
# 🎨 إعداد التنسيق والألوان العامة
# ==============================
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = '#f8fafc'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['axes.edgecolor'] = '#94a3b8'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelcolor'] = '#0f172a'
plt.rcParams['xtick.color'] = '#334155'
plt.rcParams['ytick.color'] = '#334155'
plt.rcParams['font.size'] = 12

# ==============================
# ⚙️ إنشاء إشارة ECG تجريبية
# ==============================
fs = 360  # تردد العينة (Hz)
t = np.linspace(0, 2, 2*fs)
# توليد إشارة ECG مع ضوضاء خفيفة
ecg = 1.5*np.sin(2*np.pi*1.2*t) + 0.25*np.sin(2*np.pi*20*t) + 0.1*np.random.randn(len(t))

# ==============================
# 📈 الرسم الأول: إشارة ECG ثابتة
# ==============================
plt.figure(figsize=(12, 5))
plt.plot(t, ecg, color='#2563eb', linewidth=2.2, label='ECG Signal')

plt.title('Enhanced ECG Signal Visualization', fontsize=16, color='#1e293b', pad=20)
plt.xlabel('Time (seconds)', fontsize=13)
plt.ylabel('Amplitude (mV)', fontsize=13)
plt.legend(facecolor='#e2e8f0', edgecolor='#94a3b8', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# ==============================
# 📊 الرسم الثاني: توزيع الإشارة (Histogram)
# ==============================
plt.figure(figsize=(6, 4))
plt.hist(ecg, bins=30, color='#38bdf8', edgecolor='#0c4a6e', alpha=0.9)
plt.title('ECG Signal Distribution', fontsize=14, color='#1e293b')
plt.xlabel('Amplitude (mV)')
plt.ylabel('Frequency')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ==============================
# 💓 الرسم الثالث: ECG Animation (إشارة متحركة)
# ==============================
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 2)
ax.set_ylim(min(ecg) - 0.5, max(ecg) + 0.5)
line, = ax.plot([], [], color='#16a34a', linewidth=2.5)
ax.set_title('Real-time ECG Simulation', fontsize=15, color='#1e293b')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (mV)')
ax.grid(True, linestyle='--', alpha=0.4)

# تهيئة البيانات المتحركة
def init():
    line.set_data([], [])
    return line,

# تحديث الإشارة في كل فريم
def update(frame):
    start = frame
    end = frame + 100
    if end > len(t):
        end = len(t)
    line.set_data(t[start:end], ecg[start:end])
    return line,

# إنشاء الأنيميشن
ani = animation.FuncAnimation(
    fig, update, frames=np.arange(0, len(t)-100), init_func=init,
    blit=True, interval=25, repeat=True
)

plt.tight_layout()
plt.show()
