from flask import Flask, render_template

app = Flask(__name__)

mock_data = {
    "hero_slides": [
        {
            "title": "사이판 5,6일 #켄싱턴호텔 #오션뷰\n호캉스 #1일\n2식/3식 호텔식",
            "hashtags": ["#온천호텔", "#교토", "#오사카", "#유니버셜스튜디오"],
            "bg_image": "https://images.unsplash.com/photo-1540206351-d6465b3ac5c1?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80",
            "cards": [
                {
                    "title": "홋카이도 4일 #호텔 확정",
                    "original_price": "800,000",
                    "price": "680,000",
                    "discount": "15%",
                    "image": "https://images.unsplash.com/photo-1542051841857-5f90071e7989?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
                    "tags": ["#일본", "#온천"]
                },
                {
                    "title": "도쿄 3일 #시내관광",
                    "original_price": "600,000",
                    "price": "450,000",
                    "discount": "25%",
                    "image": "https://images.unsplash.com/photo-1503899036084-c55cdd92da26?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
                    "tags": ["#도쿄", "#쇼핑"]
                }
            ]
        },
        {
            "title": "다낭/호이안 4/5일 #바나힐\n#골든브릿지 #콩카페",
            "hashtags": ["#베트남", "#다낭", "#가족여행", "#휴양"],
            "bg_image": "https://images.unsplash.com/photo-1552465011-b4e21bf6e79a?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80",
            "cards": [
                {
                    "title": "다낭 4일 #풀빌라",
                    "original_price": "900,000",
                    "price": "750,000",
                    "discount": "16%",
                    "image": "https://images.unsplash.com/photo-1565035010268-a3816f98589a?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
                    "tags": ["#휴양", "#수영장"]
                },
                {
                    "title": "나트랑 5일 #빈펄랜드",
                    "original_price": "850,000",
                    "price": "690,000",
                    "discount": "18%",
                    "image": "https://images.unsplash.com/photo-1565636291755-72a3707e5b92?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
                    "tags": ["#나트랑", "#테마파크"]
                }
            ]
        },
        {
            "title": "유럽의 낭만, 이탈리아 일주\n8/9일 #가성비여행",
            "hashtags": ["#유럽", "#이탈리아", "#로마", "#피렌체"],
            "bg_image": "https://images.unsplash.com/photo-1523906834658-6e24ef2386f9?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80",
            "cards": [
                {
                    "title": "이탈리아 9일 #완전일주",
                    "original_price": "2,500,000",
                    "price": "2,100,000",
                    "discount": "16%",
                    "image": "https://images.unsplash.com/photo-1516483638261-f4dbaf036963?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
                    "tags": ["#역사", "#문화"]
                },
                {
                    "title": "스위스/이탈리아 10일",
                    "original_price": "3,200,000",
                    "price": "2,890,000",
                    "discount": "10%",
                    "image": "https://images.unsplash.com/photo-1527668752968-14dc70a27c95?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
                    "tags": ["#알프스", "#자연"]
                }
            ]
        }
    ],
    "icons": [
        {"label": "골프여행", "icon": "fa-solid fa-golf-ball-tee"},
        {"label": "허니문", "icon": "fa-solid fa-heart"},
        {"label": "휴양지", "icon": "fa-solid fa-umbrella-beach"},
        {"label": "동남아 여행", "icon": "fa-brands fa-youtube"},
        {"label": "패키지", "icon": "fa-solid fa-suitcase"},
        {"label": "크루즈", "icon": "fa-solid fa-ship"},
        {"label": "해외숙소", "icon": "fa-solid fa-hotel"},
        {"label": "항공예약", "icon": "fa-solid fa-plane"},
        {"label": "여행의 발견", "icon": "fa-brands fa-instagram"},
        {"label": "여행 LIVE", "icon": "fa-solid fa-life-ring"}
    ],
    "products_a": [
        {
            "title": "홋카이도 4일 #호텔 확정 #온천 호텔 숙박 #오타루 산책",
            "original_price": "800,000",
            "price": "680,000",
            "discount": "15%",
            "image": "https://images.unsplash.com/photo-1542051841857-5f90071e7989?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#일본", "#온천", "#가족여행"]
        },
        {
            "title": "튀르키예(터키) 일주 8~10일 #가성비 여행 #터키국내선1회 #터키음식3대",
            "original_price": "2,000,000",
            "price": "1,780,000",
            "discount": "11%",
            "image": "https://images.unsplash.com/photo-1524231757912-21f4fe3a7200?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#터키", "#역사", "#문화"]
        },
        {
            "title": "이탈리아 일주 8/9일 #가성비여행",
            "original_price": "2,100,000",
            "price": "1,799,000",
            "discount": "14%",
            "image": "https://images.unsplash.com/photo-1523906834658-6e24ef2386f9?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#유럽", "#이탈리아", "#낭만"]
        },
        {
            "title": "시드니 5~8일 #뜨거운 여름에 만나는 시원한 조개꽃! #블루마운틴 #포트",
            "original_price": "450,000",
            "price": "354,700",
            "discount": "21%",
            "image": "https://images.unsplash.com/photo-1506973035872-a4ec16b8e8d9?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#호주", "#시드니", "#자연"]
        }
    ],
    "promo": {
        "title": "매일이 즐겁고 풍요로운 동남아의 지상 낙원으로",
        "keywords": ["여유있는 힐링, 일본", "동남아의 지상낙원료", "여행 LIVE", "생생한 정보"],
        "bg_image": "https://images.unsplash.com/photo-1537996194471-e657df975ab4?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80",
        "card": {
             "title": "튀르키예(터키) 일주 8~10일 #가성비 여행 #터키국내선...",
             "desc": "전일정 5성급호텔 숙박, 밸리댄스 포함, 사프란볼루 등 관광 포함, 알차게 다녀올 수 있는 상품입니다.",
             "original_price": "2,000,000",
             "price": "1,780,000",
             "discount": "11%",
             "image": "https://images.unsplash.com/photo-1527838832700-5059252407fa?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
        }
    },
    "products_b": [
        {
            "title": "푸켓 4일 #풀빌라 확정 #요트 투어 #스파 마사지",
            "original_price": "900,000",
            "price": "680,000",
            "discount": "24%",
            "image": "https://images.unsplash.com/photo-1589394815804-964ed0be2eb5?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#태국", "#휴양", "#풀빌라"]
        },
        {
            "title": "다낭/호이안 4/5일 #바나힐 #골든브릿지 #콩카페",
            "original_price": "500,000",
            "price": "399,000",
            "discount": "20%",
            "image": "https://images.unsplash.com/photo-1552465011-b4e21bf6e79a?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#베트남", "#다낭", "#가족"]
        },
        {
            "title": "보라카이 4/5일 #화이트비치 #세일링보트 #호핑투어",
            "original_price": "600,000",
            "price": "450,000",
            "discount": "25%",
            "image": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#필리핀", "#보라카이", "#바다"]
        },
        {
            "title": "코타키나발루 5일 #반딧불투어 #선셋 #호핑투어",
            "original_price": "550,000",
            "price": "420,000",
            "discount": "23%",
            "image": "https://images.unsplash.com/photo-1573455494060-c5595004fb6c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#말레이시아", "#석양", "#자연"]
        }
    ]
}

@app.route('/')
def index():
    return render_template('index.html', data=mock_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7879, debug=True)
