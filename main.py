import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Streamlit 앱 제목
st.title("외계 행성 탐사: 림 다크닝 포함 광도 변화 시뮬레이션")

# 설명
st.write("""
이 앱은 외계 행성이 항성을 통과할 때 발생하는 광도 변화를 시뮬레이션합니다.
림 다크닝 효과를 포함하여 더 현실적인 광도 변화를 확인할 수 있습니다.
항성과 행성의 반지름 및 림 다크닝 계수를 조정하세요.
""")

# 입력 슬라이더
st.header("입력 매개변수")
star_radius = st.slider("항성 반지름 (태양 반지름 단위, R☉)", 
                        min_value=0.1, max_value=2.0, value=1.0, step=0.1)
planet_radius = st.slider("행성 반지름 (목성 반지름 단위, R_J)", 
                          min_value=0.1, max_value=2.0, value=1.0, step=0.1)
limb_darkening_coeff = st.slider("림 다크닝 계수 (u)", 
                                 min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# 반지름 단위 변환 (1 목성 반지름 ≈ 0.10045 태양 반지름)
planet_radius_solar = planet_radius * 0.10045

# 림 다크닝 포함 광도 계산 함수
def limb_darkening_intensity(mu, u):
    """선형 림 다크닝 모델: I(mu) = 1 - u(1 - mu)"""
    return 1 - u * (1 - mu)

def transit_flux(d, Rp, Rs, u):
    """행성 통과 시 상대 광도 계산 (림 다크닝 포함)
    d: 행성 중심과 항성 중심 간 거리 (항성 반지름 단위)
    Rp: 행성 반지름 (항성 반지름 단위)
    Rs: 항성 반지름 (항성 반지름 단위, 1로 정규화)
    u: 림 다크닝 계수
    """
    if d >= 1 + Rp:  # 행성이 항성 밖에 있음
        return 1.0
    elif d <= Rp - 1:  # 항성이 행성에 완전히 가려짐 (불가능한 경우)
        return 0.0
    
    # 항성 표면의 총 광도 (림 다크닝 포함)
    total_flux, _ = quad(lambda mu: 2 * np.pi * mu * limb_darkening_intensity(mu, u), 0, 1)
    
    # 행성이 가리는 영역 계산
    if d <= 1 - Rp:  # 행성이 항성 내부에 완전히 있음
        blocked_area = np رژیم

System: ### 수정된 코드 (림 다크닝 포함)

림 다크닝 효과를 포함한 Streamlit 앱 코드를 완성하겠습니다. 이전 코드의 구조를 유지하면서 림 다크닝 효과를 반영한 광도 변화 계산을 추가합니다. 림 다크닝은 선형 모델 \( I(\mu) = 1 - u(1 - \mu) \)를 사용하며, 행성 통과 시 가려진 항성 표면의 광도를 적분하여 상대 광도를 계산합니다. 사용자가 항성 반지름, 행성 반지름, 림 다크닝 계수를 조정할 수 있도록 인터페이스를 제공합니다.

### 코드

<xaiArtifact artifact_id="0e1ec580-1a11-47bb-b660-fa08c05978c0" artifact_version_id="10f5d29d-3877-4eec-8d8f-8a0a7fd5855e" title="transit_simulation_with_limb_darkening.py" contentType="text/python">
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Streamlit 앱 제목
st.title("외계 행성 탐사: 림 다크닝 포함 광도 변화 시뮬레이션")

# 설명
st.write("""
이 앱은 외계 행성이 항성을 통과할 때 발생하는 광도 변화를 시뮬레이션합니다.
림 다크닝 효과를 포함하여 더 현실적인 광도 변화를 확인할 수 있습니다.
항성과 행성의 반지름 및 림 다크닝 계수를 조정하세요.
""")

# 입력 슬라이더
st.header("입력 매개변수")
star_radius = st.slider("항성 반지름 (태양 반지름 단위, R☉)", 
                        min_value=0.1, max_value=2.0, value=1.0, step=0.1)
planet_radius = st.slider("행성 반지름 (목성 반지름 단위, R_J)", 
                          min_value=0.1, max_value=2.0, value=1.0, step=0.1)
limb_darkening_coeff = st.slider("림 다크닝 계수 (u)", 
                                 min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# 반지름 단위 변환 (1 목성 반지름 ≈ 0.10045 태양 반지름)
planet_radius_solar = planet_radius * 0.10045

# 림 다크닝 포함 광도 계산 함수
def limb_darkening_intensity(mu, u):
    """선형 림 다크닝 모델: I(mu) = 1 - u(1 - mu)"""
    return 1 - u * (1 - mu)

def transit_flux(d, Rp, Rs, u):
    """행성 통과 시 상대 광도 계산 (림 다크닝 포함)
    d: 행성 중심과 항성 중심 간 거리 (항성 반지름 단위)
    Rp: 행성 반지름 (항성 반지름 단위)
    Rs: 항성 반지름 (항성 반지름 단위, 1로 정규화)
    u: 림 다크닝 계수
    """
    # 항성 표면의 총 광도 (림 다크닝 포함)
    total_flux, _ = quad(lambda mu: 2 * np.pi * mu * limb_darkening_intensity(mu, u), 0, 1)
    
    if d >= 1 + Rp:  # 행성이 항성 밖에 있음
        return 1.0
    elif d <= Rp - 1:  # 항성이 행성에 완전히 가려짐 (불가능한 경우)
        return 0.0
    
    # 가려진 영역의 광도 계산
    def integrand(mu):
        return limb_darkening_intensity(mu, u) * 2 * np.arccos((d**2 + mu**2 - Rp**2) / (2 * d * mu))
    
    if d <= 1 - Rp:  # 행성이 항성 내부에 완전히 있음
        blocked_flux, _ = quad(integrand, Rp - d, 1)
    else:  # 부분 통과
        r1 = max(0, d - Rp)
        r2 = min(1, d + Rp)
        blocked_flux, _ = quad(integrand, r1, r2)
    
    return 1 - blocked_flux / total_flux

# 시간 배열 생성 (정규화된 시간, -1.5 ~ 1.5)
time = np.linspace(-1.5, 1.5, 100)
# 행성 중심과 항성 중심 간 거리 (항성 반지름 단위)
d = np.abs(time) * (1 + planet_radius_solar / star_radius)

# 광도 변화 계산
flux = np.array([transit_flux(di, planet_radius_solar / star_radius, 1.0, limb_darkening_coeff) for di in d])

# 최대 광도 감소 비율 계산 (림 다크닝 고려하지 않은 단순 모델로 근사)
max_flux_drop = (planet_radius_solar / star_radius) ** 2 * 100  # 퍼센트 단위

# 그래프 생성
fig, ax = plt.subplots()
ax.plot(time, flux, color='blue', label='상대 광도 (림 다크닝 포함)')
ax.set_xlabel('정규화된 시간')
ax.set_ylabel('상대 광도 (F/F₀)')
ax.set_title('행성 통과에 따른 항성 광도 변화')
ax.grid(True)
ax.legend()

# 그래프 표시
st.pyplot(fig)

# 결과 출력
st.header("결과")
st.write(f"**항성 반지름**: {star_radius:.2f} R☉")
st.write(f"**행성 반지름**: {planet_radius:.2f} R_J ({planet_radius_solar:.3f} R☉)")
st.write(f"**림 다크닝 계수**: {limb_darkening_coeff:.2f}")
st.write(f"**최대 광도 감소 (근사)**: {max_flux_drop:.3f}%")

# 추가 정보
st.write("""
### 참고
- **림 다크닝**: 항성 표면의 밝기가 중심에서 가장자리로 갈수록 감소하는 효과를 반영합니다.
- 선형 � MagnetoPyrolysis 림 다크닝 모델 사용: I(μ) = 1 - u(1 - μ).
- 광도 감소는 행성이 가리는 항성 표면의 밝기를 적분하여 계산됩니다.
- 최대 광도 감소는 단순 모델로 근사한 값이며, 실제 값은 림 다크닝으로 인해 약간 다를 수 있습니다.
- 시간은 정규화된 단위로, 실제 통과 시간은 궤도 주기와 항성 크기에 따라 달라집니다.
""")
