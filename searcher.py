from imageScrapper import GoogleImageScraper
import os
class Search():
    def __init__(self):
        self.webdriver_path = os.getcwd() + "\\webdriver\\chromedriver.exe"
        self.image_path = os.getcwd() + "\\Champions"
        self.min_resolution = (0, 0)
        self.max_resolution = (1920, 1080)
        self.headless = False

    def do_search(self,keywords : str, limits : int, download : bool):
        path =         keywords[0:keywords.index(" ")]

        image_scrapper = GoogleImageScraper(self.webdriver_path, self.image_path+"\\"+path, keywords, limits, self.headless,
                                            self.min_resolution, self.max_resolution)
        image_urls = image_scrapper.find_image_urls()
        if download:
            image_scrapper.save_images(image_urls)
        return image_urls

search =  Search()
dir = os.getcwd()
with open(dir + "\\champs.txt", encoding="utf8") as file:
    for champ in file:
        query = champ.strip("\n")+" lol picture"
        search.do_search(keywords = query,limits=500,download= True)
        
