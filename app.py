
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pytz

load_dotenv()

# 티커 목록
TICKERS = {
    "나스닥": "^IXIC",
    "S&P 500": "^GSPC",
    "다우존스": "^DJI",
    "달러/원": "KRW=X",
    "미국 10년물 금리": "^TNX",
}

NEWS_URLS = [
    "https://finance.yahoo.com/topic/stock-market-news/",
    "https://finance.yahoo.com/topic/economic-news/",
]

BRIEFING_PROMPT = """아래는 오늘 미국 증시 관련 주요 뉴스 헤드라인이야.
이걸 바탕으로 출근길에 읽기 좋은 간결한 시장 브리핑을 한국어로 작성해줘.

작성 형식:
1. 오늘 시장 분위기 (2~3문장, 자연스러운 구어체로)
2. 오늘 핵심 이슈 3가지 (간단히 bullet로)
3. 주목할 섹터 또는 테마
4. 오늘 조심해야 할 리스크

뉴스:
{news}
"""

TRANSLATE_PROMPT = """아래 영어 뉴스 헤드라인을 자연스러운 한국어로 번역해줘.
번역문만 출력하고 다른 말은 하지 마.

{headline}
"""

CHAT_SYSTEM = """너는 미국 주식 시장을 잘 아는 친한 선배 투자자야.
반말로 편하게, 하지만 정확하게 답변해줘.
투자 추천은 하지 말고 정보 위주로 설명해줘.
"""


def init_page():
    st.set_page_config(
        page_title="미국 주식 모닝 브리핑",
        page_icon="📈",
        layout="wide"
    )
    st.title("📈 미국 주식 모닝 브리핑")
    now_kr = datetime.now(pytz.timezone("Asia/Seoul"))
    st.caption(f"한국 시간 기준  |  {now_kr.strftime('%Y년 %m월 %d일 %H:%M')}")
    st.sidebar.title("설정")


def select_model(temperature=0):
    models = ("gpt-5.5", "gpt-5.4-mini")
    model = st.sidebar.radio("모델 선택", models)
    return ChatOpenAI(temperature=temperature, model=model)

def select_back():
    theme = st.sidebar.radio("테마 선택", ["라이트모드", "다크모드"])

    if theme == "다크모드":
        st.markdown("""
            <style>
            .stApp {
                background-color: #0E1117;
                color: #FFFFFF;
                /* 사이드바 배경 */
                [data-testid="stSidebar"] {
                    background-color: #1E2130 !important;
                }
                /* 사이드바 글자 */
                [data-testid="stSidebar"] * {
                    color: #FFFFFF !important;
                }
                /* 티커 라벨 */
                [data-testid="stMetricLabel"] {
                    color: #FFFFFF !important;
                }
                /* 지수 숫자 */
                [data-testid="stMetricValue"] {
                    color: #FFFFFF !important;
                }
                /* 등락% */
                [data-testid="stMetricDelta"] {
                    filter: brightness(1.5);
                }
            }
            </style>
        """, unsafe_allow_html=True)
    else:  # ← 라이트모드일 때도 명시적으로 흰 배경 지정
        st.markdown("""
            <style>
            .stApp {
                background-color: #FFFFFF;
                color: #000000;
            }
            </style>
        """, unsafe_allow_html=True)

def get_market_data():
    results = {}
    for name, ticker in TICKERS.items():
        hist = yf.Ticker(ticker).history(period="2d") # 최근 2일치 주가 데이터 확인
        if len(hist) >= 2:
            prev = hist["Close"].iloc[-2] # 전일 종가
            cur = hist["Close"].iloc[-1]  # 오늘 종가
            change = cur - prev
            pct = (change / prev) * 100
            results[name] = {"price": cur, "change": change, "pct": pct}
        else:
            results[name] = None
    return results

def show_market_data(market_data):
    st.subheader("시장 현황")
    cols = st.columns(len(TICKERS))
    for col, (name, data) in zip(cols, market_data.items()):
        with col:
            if data:
                sign = "+" if data["change"] >= 0 else "" # 양수면 Sign +세팅
                st.metric(
                    label=name,                       # 티커
                    value=f"{data['price']:,.2f}",    # 가격(소수점 둘째)
                    delta=f"{sign}{data['pct']:.2f}%" # 등호 + 값(소수점 둘째) + %
                )
            else:
                st.metric(label=name, value="데이터 없음")


def get_news():
    headlines = []
    for url in NEWS_URLS:
        try:
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            soup = BeautifulSoup(res.text, "html.parser")
            for tag in soup.find_all("h3", limit=10):
                text = tag.get_text(strip=True)
                if len(text) > 20:
                    headlines.append(text)
        except Exception:
            pass
    return headlines[:15]

def show_ai_briefing(llm, news_list):
    st.subheader("오늘의 브리핑")
    if not news_list:
        st.warning("뉴스를 불러오지 못했어요. 잠시 후 다시 시도해주세요.")
        return

    news_text = "\n".join([f"- {n}" for n in news_list])
    prompt = ChatPromptTemplate.from_messages([("user", BRIEFING_PROMPT)])
    chain = prompt | llm | StrOutputParser()

    with st.spinner("브리핑 작성 중..."):
        st.write_stream(chain.stream({"news": news_text}))

def show_news(llm, news_list):
    st.subheader("오늘의 주요 뉴스")
    if not news_list:
        st.info("뉴스를 불러오지 못했습니다.")
        return

    for i, headline in enumerate(news_list, 1):
        with st.expander(f"{i}. {headline}"):
            with st.spinner("번역 중..."):
                translated = translate_headline(llm, headline)
            st.markdown(f"**🇰🇷 한국어:** {translated}")

def translate_headline(llm, headline):
    prompt = ChatPromptTemplate.from_messages([("user", TRANSLATE_PROMPT)])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"headline": headline})



def show_chatbot(llm):
    st.subheader("종목 분석 챗봇")
    st.caption("궁금한 종목이나 시장 상황을 편하게 물어보세요")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("예: 엔비디아 요즘 어때?ㅠㅠ / 나스닥 왜 떨어진 거야? ㅠㅠ")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        prompt = ChatPromptTemplate.from_messages([
            ("system", CHAT_SYSTEM),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}")
        ])
        chain = prompt | llm | StrOutputParser()
        history = [(m["role"], m["content"]) for m in st.session_state.chat_history[:-1]]

        with st.chat_message("assistant"):
            response = st.write_stream(chain.stream({
                "history": history,
                "input": user_input
            }))

        st.session_state.chat_history.append({"role": "assistant", "content": response})


def main():
    init_page()
    llm = select_model()
    back = select_back()

    tab1, tab2, tab3 = st.tabs(["시장 현황 & 브리핑", "뉴스 헤드라인", "종목 분석 챗봇"])

    with tab1:
        with st.spinner("시장 데이터 불러오는 중..."):
            market_data = get_market_data()
        show_market_data(market_data)
        st.divider()
        news_list = get_news()
        show_ai_briefing(llm, news_list)

    with tab2:
        news_list = get_news()
        show_news(llm, news_list)

    with tab3:
        show_chatbot(llm)

main()
