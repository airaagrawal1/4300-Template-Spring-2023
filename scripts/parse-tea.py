# import libraries 
from bs4 import BeautifulSoup
import requests 
import json
import re 

url = 'https://ratetea.com/styles.php'
tea_base_url = "https://ratetea.com"
req = requests.get(url)
content = req.text 
soup = BeautifulSoup(content, 'html.parser')

tea_data = []
tea_type = ["Pure Tea (Camellia sinensis)", "Flavored Tea", "Herbal Tea"]

def get_tea_data(base_url, tea_url): 
    tea_content = requests.get(tea_url).text
    tea_soup = BeautifulSoup(tea_content, "html.parser")
    print(tea_url)

    # about text
    left_data = tea_soup.select('.left_of_wide_cell')[0]
    about_text = ""
    for l_tag in left_data: 
        if l_tag.name == "h2" or l_tag.name == "div" or l_tag.name == "h3":
            continue
        about_text += l_tag.getText()
        
    about_text = re.sub("\\s+", " ", about_text)

    # average rating 
    rating_data = tea_soup.select("#recent-reviews")
    avg_rating = 0
    if rating_data: 
        rating_data = rating_data[0]
        rating_data = rating_data.find_all_next("div", attrs={'style': "float: right; height: 20px; padding: 2px; font-weight: bold;"})
        for rating in rating_data: 
            rating_text = rating.getText()
            divide_index = rating_text.index("/")
            rate = float(rating_text[:divide_index].strip())
            avg_rating += rate
        avg_rating /= len(rating_data)
    
    # reviews 
    review_data = tea_soup.select("#recent-reviews")
    reviews = []
    if review_data: 
        review_data = review_data[0]
        review_data = review_data.find_all_next("a")
        review_links = []
        for review in review_data:
            review_id = review.get('id')
            if (review_id):
                ri = review_id.index('g')
                review_links.append(review_id[ri+1:])
            else: 
                continue
        for link in review_links:
            review = ""
            review_url = base_url + "/review/" + link 
            review_content = requests.get(review_url).text
            review_soup = BeautifulSoup(review_content, "html.parser")
            review_data = review_soup.select('.article p')[0]
            for r_tag in review_data: 
                review += r_tag.getText()
            reviews.append(re.sub("\\s+", " ", review))

    # top rated tea brands
    brand_data = tea_soup.select("#top-rated")
    brands = []
    if brand_data:
        brand_data = brand_data[0]
        brand_data = brand_data.find_all_next("td")
        get_next = False
        for td in brand_data: 
            if td.getText() == "Brand:":
                get_next = True 
                continue
            
            if get_next: 
                brands.append(td.getText())
                get_next = False

    return about_text, avg_rating, reviews, brands


tea_boxes = soup.select('.big_list_nm')
for box in tea_boxes: 
    if box.contents == []: 
        continue
    for box_contents in box.contents: 
        if box_contents.getText() in tea_type:
            current_tea_type = box_contents.getText()
            continue
        if box_contents.getText() == "": 
            continue

        # go into the link and get the tea specific data 
        tea_url = tea_base_url + box_contents.find('a').get('href')
        about_text, avg_rating, reviews, brands = get_tea_data(tea_base_url, tea_url)
        
        tea_data.append({
            "tea_type": current_tea_type, 
            "tea_category": box_contents.getText(),
            "about": about_text, 
            "avg_rating": avg_rating, 
            "reviews": reviews, 
            "top_rated_brands": brands
        })

tea_json = { "data": tea_data }
json_data = json.dumps(tea_json)

with open("tea_data.json", "w") as file: 
    file.write(json_data)