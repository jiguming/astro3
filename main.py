import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Streamlit 앱 제목
st.title("외계 행성 탐사: 점진적 광도 변화 시뮬레이션")

# 설명
st.write("""
이 앱은 외계 행성이 항성을 통과할 때 발생하는 광도 변화를 시뮬레이션합니다.
행성이 항성 앞을 지나면서 초반에 밝기가 서서히 감소하고(ingress), 최대 감소 후 서서히 회복되는(egress) 과정을 반영합니다.
항성과 행성의 반지름을 조정하여 광도 변화 곡선을 확인하세요.
""")

# 입력 슬라이더
st.header("입력 매개변수")
star_radius = st.slider("항성 반지름 (태양 반지름 단위, R☉)", 
                        min_value=0.1, max_value=2.0, value=1.0, step=0.1)
planet_radius = st.slider("행성 반지름 (목성 반지름 단위, R_J)", 
                          min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# 반지름 단위 변환 (1 목성 반지름 ≈ 0.10045 태양 반지름)
planet_radius_solar = planet_radius * 0.10045

# 행성과 항성의 겹치는 면적 계산 함수
def occultation_area(d, Rp, Rs):
    """행성과 항성의 겹치는 면적 계산
    d: 행성 중심과 항성 중심 간 거리 (항성 반지름 단위)
    Rp: 행성 반지름 (항성 반지름 단위)
    Rs: 항성 반지름 (항성 반지름 단위, 1로 정규화)
    """
    if d >= 1 + Rp:  # 행성이 항성 밖에 있음
        return 0.0
    if d <= Rp - 1:  # 항성이 행성에 완전히 가려짐 (불가능)
        return np.pi
    if d <= 1 - Rp:  # 행성이 항성 내부에 완전히 있음
        return np.pi * Rp**2
    
    # 부분 통과 (ingress/egress)
    d1 = (1 + Rp**2 - d**2) / (2 * Rp)
    d2 = (d**2 + 1 - Rp**2) / (2 * d)
    term1 = Rp**2 * np.arccos(np.clip(d1, -1, 1))
    term2 = np.arccos(np.clip(d2, -1, 1))
    term3 = 0.5 * np.sqrt(max(0, (1 + Rp - d) * (1 + Rp + d) * (d + Rp - 1) * (d - Rp + 1)))
    return term1 + term2 - term3

# 광도 변화 계산 함수
def transit_flux(d, Rp, Rs):
    """행성 통과 시 상대 광도 계산 (림 다크닝 무시)
    d: 행성 중심과 항성 중심 간 거리 (항성 반지름 단위)
    Rp: 행성 반지름 (항성 반지름 단위)
    Rs: 항성 반지름 (항성 반지름 단위, 1로 정규화)
    """
    blocked_area = occultation_area(d, Rp, Rs)
    total_area = np.pi * Rs**2
    return 1 - blocked_area / total_area

# 시간 배열 생성 (정규화된 시간, -1.5 ~ 1.5)
time = np.linspace(-1.5, 1.5, 200)
# 행성 중심과 항성 중심 간 거리 (항성 반지름 단위)
d = np.abs(time) * (1 + planet_radius_solar / star_radius)

# 광도 변화 계산
flux = np.array([transit_flux(di, planet_radius_solar / star_radius, 1.0) for di in d])

# 최대 광도 감소 비율 계산
max_flux_drop = (planet_radius_solar / star_radius) ** 2 * 100  # 퍼센트 단위

# 그래프 생성
fig, ax = plt.subplots()
ax.plot(time, flux, color='blue', label='상대 광도')
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
st.write(f"**최대 광도 감소**: {max_flux_drop:.3f}%")

# 추가 정보
st.write("""
### 참고
- **점진적 변화**: 행성이 항성 디스크에 진입(ingress)하고 빠져나가는(egress) 과정에서 광도가 서서히 변합니다.
- 광도 감소는 행성이 가리는 항성 면적의 비율로 계산됩니다.
- 림 다크닝 효과는 무시되어 항성 표면의 밝기가 균일하다고 가정합니다.
- 시간은 정규화된 단위로, 실제 통과 시간은 궤도 주기와 항성 크기에 따라 달라집니다.
""")
