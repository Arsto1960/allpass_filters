import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Page Config ---
st.set_page_config(
    page_title="Allpass & Warping Explorer",
    page_icon="🌀",
    layout="wide"
)

# --- CSS Styling ---
st.markdown("""
<style>
    .metric-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# st.title("🌀 Allpass Filters & Frequency Warping")
st.markdown("### 🌀 Allpass Filters & Frequency Warping")
st.markdown("""
Explore how **Allpass Filters**—which pass all frequencies with magnitude 1 but alter phase—are used for **Fractional Delays** and **Psychoacoustic Frequency Warping** (Bark Scale).
""")

# --- Helper Functions from Text ---
def get_allpass_coeffs(a):
    """Returns B, A for H(z) = (z^-1 - a*) / (1 - a*z^-1)"""
    # Note: Text defines H(z) = (z^-1 - conj(a)) / (1 - a*z^-1)
    # Scipy lfilter expects A[0]=1. 
    # Numerator B: [-conj(a), 1] corresponds to -a* + z^-1
    # Denominator A: [1, -a] corresponds to 1 - a*z^-1
    B = [-np.conj(a), 1]
    A = [1, -a]
    return B, A

def warping_phase(w, a):
    """Calculates warped frequency phase per Eq 9.1"""
    # Eq 9.1: phi(w) = -w - 2*atan( (r*sin(w-theta)) / (1-r*cos(w-theta)) )
    theta = np.angle(a)
    r = np.abs(a)
    
    # Avoid division by zero
    denom = 1 - r * np.cos(w - theta)
    denom[denom == 0] = 1e-10
    
    term = (r * np.sin(w - theta)) / denom
    phi = -w - 2 * np.arctan(term)
    return phi

def iir_fractional_delay(tau):
    """Thiran Allpass Interpolator from text section 9.2.3"""
    L = int(tau) + 1
    n = np.arange(0, L+1) # 0 to L
    
    # Text calculation logic
    # a_k = (-1)^k * choose(L, k) * prod(...) 
    # Using the specific Python code provided in text:
    # a = cumprod( (L-n)/(n+1) * (L-n-tau)/(n+1+tau) ) 
    # Note: The text code snippet had specific ranges. Re-implementing closely.
    
    # Start with a_0 = 1
    a_coeffs = [1.0]
    curr = 1.0
    for k in range(L): 
        # k goes 0 to L-1. 
        # Formula term: (L-k)/(k+1) * (L-k-tau)/(k+1+tau)
        term = ((L - k) * (L - k - tau)) / ((k + 1) * (k + 1 + tau))
        curr *= term
        a_coeffs.append(curr)
        
    a = np.array(a_coeffs)
    b = a[::-1] # Flip for numerator
    return b, a

def fir_fractional_delay(d, L=10):
    """Windowed Sinc fractional delay"""
    # Center the filter
    n0 = L // 2
    n = np.arange(L)
    
    # Sine Window (from text)
    w = np.sin(np.pi / L * (n + 0.5))
    
    # Shifted Sinc
    # h(n) = sinc(-d + n - n0) * w(n)
    # Note: d here is the FRACTIONAL part + Integer Shift desired. 
    # The text says "sinc(-d + n - n0)". If we want total delay 'delay', 
    # and we center at n0, then d in the formula represents the delay relative to n0.
    
    # Let's say target total delay is 'd_total'.
    # We want peak at d_total.
    # The sinc argument is (n - d_total).
    # Text uses: sinc(-d + n - n0). 
    
    h = np.sinc(n - d) * w
    return h

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "1️⃣ Allpass Fundamentals",
    "2️⃣ Fractional Delays",
    "3️⃣ Frequency Warping (Bark)"
])

# ==============================================================================
# TAB 1: ALLPASS FUNDAMENTALS
# ==============================================================================
with tab1:
    # st.header("1. First-Order Allpass Filter")
    st.markdown(r"""
    The fundamental building block is the first-order allpass filter:
    $$ H_{ap}(z) = \frac{z^{-1} - \bar{a}}{1 - a z^{-1}} $$
    * **Magnitude:** Constant 1 everywhere ($|H(e^{j\Omega})|=1$).
    * **Pole/Zero:** Pole at $a$, Zero at $1/\bar{a}$ (Conjugate Reciprocal positions).
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Design")
        a_real = st.slider("Coefficient 'a' (Real part)", -0.9, 0.9, 0.5, 0.1)
        st.caption("Change 'a' to see the Pole/Zero move. For stability, $|a| < 1$.")
        
        B, A = get_allpass_coeffs(a_real)
        
    with col2:
        # --- Plots ---
        fig1, ax = plt.subplots(1, 2, figsize=(10, 4))
        fig1.patch.set_alpha(0)
        
        # 1. Pole-Zero Plot
        circle = plt.Circle((0,0), 1, color='green', fill=False, linestyle='--')
        ax[0].add_patch(circle)
        
        # Calculate P/Z
        z, p, k = signal.tf2zpk(B, A)
        ax[0].scatter(np.real(p), np.imag(p), marker='x', s=100, color='red', label='Pole')
        ax[0].scatter(np.real(z), np.imag(z), marker='o', s=100, facecolors='none', edgecolors='blue', label='Zero')
        
        ax[0].axvline(0, color='gray', lw=0.5)
        ax[0].axhline(0, color='gray', lw=0.5)
        ax[0].set_xlim(-2.5, 2.5)
        ax[0].set_ylim(-1.5, 1.5)
        ax[0].set_aspect('equal')
        ax[0].set_title("Pole-Zero Plot")
        ax[0].legend(loc='lower right')
        ax[0].grid(True, alpha=0.3)
        
        # 2. Phase & Group Delay
        w, H = signal.freqz(B, A, worN=512)
        w_norm = w / np.pi
        
        # Group Delay
        w_gd, gd = signal.group_delay((B, A), w=512)
        
        ax[1].plot(w_norm, gd, 'b-', label='Group Delay')
        ax[1].set_title("Group Delay (Samples)")
        ax[1].set_xlabel("Normalized Freq ($\times \pi$)")
        ax[1].set_ylabel("Delay (samples)")
        ax[1].grid(True, alpha=0.3)
        
        st.pyplot(fig1)
        
        st.info(f"""
        **Observation:**
        * **Pole:** At $z = {a_real}$
        * **Zero:** At $z = {1/a_real:.2f}$
        * **Group Delay:** Notice it is NOT constant. Low frequencies might be delayed more or less than high frequencies depending on sign of $a$. This variation is what allows frequency warping!.
        """)

    with st.expander("Observation"):
            st.markdown(r"""
        * **Pole:** At $z = {a_real}$
        * **Zero:** At $z = {1/a_real:.2f}$
        * **Group Delay:** Notice it is NOT constant. Low frequencies might be delayed more or less than high frequencies depending on sign of $a$. This variation is what allows frequency warping!.
        """)

# ==============================================================================
# TAB 2: FRACTIONAL DELAYS
# ==============================================================================
with tab2:
    # st.header("2. Fractional Delays")
    st.markdown("""
    Standard digital delays are integers ($z^{-1}, z^{-2}$). To get a delay of **4.5 samples**, we need filters.
    * **FIR Approach:** Windowed Sinc function shifted by $d$.
    * **IIR Approach:** Allpass filter designed to have specific Group Delay at low freq.
    """)
    
    col_d1, col_d2 = st.columns([1, 2])
    
    with col_d1:
        desired_delay = st.slider("Target Delay (Samples)", 0.1, 10.0, 4.5, 0.1)
        
        st.subheader("FIR Settings")
        L_fir = st.slider("FIR Length (L)", 4, 30, 10)
        
    with col_d2:
        # --- Calculations ---
        
        # 1. FIR Design
        h_fir = fir_fractional_delay(desired_delay, L_fir)
        
        # 2. IIR Design
        # For IIR, we use the Thiran approximation function from text
        # Note: IIR Thiran is typically for the FRACTIONAL part. 
        # But the text example uses it for total delay e.g. 5.5.
        b_iir, a_iir = iir_fractional_delay(desired_delay)
        
        # 3. Test Signal (Step/Pulse)
        test_sig = np.zeros(30)
        test_sig[0] = 1 # Impulse
        
        # Filter
        y_fir = signal.lfilter(h_fir, 1, test_sig)
        y_iir = signal.lfilter(b_iir, a_iir, test_sig)
        
        # --- Visualization ---
        fig2, ax2 = plt.subplots(2, 1, figsize=(10, 8))
        fig2.patch.set_alpha(0)
        
        # Impulse Responses
        ax2[0].stem(y_fir, basefmt=" ", linefmt='b-', markerfmt='bo', label='FIR (Sinc)')
        ax2[0].plot(y_iir, 'r--', label='IIR (Allpass)')
        ax2[0].axvline(desired_delay, color='green', linestyle=':', lw=2, label=f'Target: {desired_delay}')
        ax2[0].set_title("Impulse Response Comparison")
        ax2[0].legend()
        ax2[0].grid(True, alpha=0.3)
        
        # Frequency Response (Magnitude)
        w, H_fir = signal.freqz(h_fir, 1, worN=512)
        w, H_iir = signal.freqz(b_iir, a_iir, worN=512)
        
        ax2[1].plot(w/np.pi, 20*np.log10(abs(H_fir)+1e-10), 'b-', label='FIR Mag')
        ax2[1].plot(w/np.pi, 20*np.log10(abs(H_iir)+1e-10), 'r--', label='IIR Mag')
        ax2[1].set_title("Magnitude Response (dB)")
        ax2[1].set_xlabel("Normalized Frequency")
        ax2[1].set_ylim(-20, 2)
        ax2[1].legend()
        ax2[1].grid(True, alpha=0.3)
        
        st.pyplot(fig2)
        
        # st.markdown(f"""
        # **Comparison:**
        # * **IIR (Allpass):** Magnitude is exactly **0 dB** (flat). Perfect allpass behavior, but phase/delay varies at high frequencies.
        # * **FIR (Windowed Sinc):** Magnitude **drops** at high frequencies (low-pass effect) because we truncated the infinite sinc. It's not a perfect allpass.
        # """)
    with st.expander("⚖️ Comparison:"):
            st.markdown(r"""
            * **IIR (Allpass):** Magnitude is exactly **0 dB** (flat). Perfect allpass behavior, but phase/delay varies at high frequencies.
            * **FIR (Windowed Sinc):** Magnitude **drops** at high frequencies (low-pass effect) because we truncated the infinite sinc. It's not a perfect allpass.
            """)
# ==============================================================================
# TAB 3: FREQUENCY WARPING
# ==============================================================================
with tab3:
    # st.header("3. Psychoacoustic Frequency Warping")
    st.markdown("""
    The human ear (Cochlea) has higher resolution at low frequencies. We can "warp" a filter's frequency axis to match this using an allpass substitution:
    $$ z^{-1} \leftarrow H_{ap}(z) $$
    This allows us to design filters that focus more taps/resolution on the bass frequencies.
    """)
    
    col_w1, col_w2 = st.columns([1, 2])
    with col_w1:
        st.subheader("Warping Config")
        # Bark calc from text: a = 0.85 approx for 32kHz
        warp_coeff = st.slider("Warping Coefficient 'a'", 0.0, 0.95, 0.5, 0.05)
        
        st.caption("""
        Positive 'a' stretches low frequencies (like Bark scale).
        Negative 'a' stretches high frequencies.
        """)
        
    with col_w2:
        # --- Warping Map Visualization ---
        w_in = np.linspace(0, np.pi, 500)
        
        # Calculate warped phase (interpreted as warped freq)
        # We assume the negative phase corresponds to frequency as per text section 9.2.5
        w_out_raw = warping_phase(w_in, warp_coeff)
        
        # Normalize to 0..pi for plotting (take absolute or handle signs carefully)
        # Text says: "interpret negative phase as normalized frequency"
        # The phase of allpass goes 0 to -pi (or -pi to -2pi depending on definition). 
        # Typically we plot -phi vs w.
        w_out = -w_out_raw
        
        # --- Plot ---
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        fig3.patch.set_alpha(0)
        
        # 1. Warping Curve
        ax3.plot(w_in/np.pi, w_out/np.pi, 'b-', linewidth=2, label=f'Allpass Map (a={warp_coeff})')
        
        # 2. Identity Line
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Linear (No Warp)')
        
        # 3. Add visual cues
        # Show where 0.1pi maps to
        test_freq = 0.1 * np.pi
        mapped_freq = -warping_phase(np.array([test_freq]), warp_coeff)[0]
        
        ax3.scatter([test_freq/np.pi], [mapped_freq/np.pi], color='red', zorder=5)
        ax3.annotate(f"Input 0.1$\pi$ $\to$ {mapped_freq/np.pi:.2f}$\pi$", 
                     xy=(test_freq/np.pi, mapped_freq/np.pi), 
                     xytext=(0.2, 0.5), arrowprops=dict(arrowstyle='->', color='red'))
        
        ax3.set_title("Frequency Mapping: Input $\Omega$ vs Warped $\Omega_{new}$")
        ax3.set_xlabel("Input Frequency ($\times \pi$)")
        ax3.set_ylabel("Warped Frequency ($\times \pi$)")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        st.pyplot(fig3)
        
        # st.markdown(f"""
        # **What this means:**
        # With $a={warp_coeff}$, the low frequencies (e.g., $0$ to $0.1\pi$) are "stretched" to occupy a larger range ($0$ to {mapped_freq/np.pi:.2f}$\pi$) in the new domain. 
        # This means any filter designed in the new domain will have **more coefficients/resolution** dedicated to that original low-frequency band.
        # """)

        with st.expander("💡What this means?"):
            st.markdown(r"""With $a={warp_coeff}$, the low frequencies (e.g., $0$ to $0.1\pi$) are "stretched" to occupy a larger range ($0$ to {mapped_freq/np.pi:.2f}$\pi$) in the new domain. 
            This means any filter designed in the new domain will have **more coefficients/resolution** dedicated to that original low-frequency band.
            """)
