##########################
### Car Prices Webscraper
#########################


#Scrapes prices of used cars in India from Cardekho.com.  
#Data used for econometric research into gasoline prices and automobile demand. 


from BeautifulSoup import BeautifulSoup
import urllib2
import csv
import re

#Get list of top cities
URL = "http://www.cardekho.com/usedCars"
soup = BeautifulSoup(urllib2.urlopen(URL).read())
links = soup.findAll("div",attrs={"id":["UsedCarsCity"],"class":["NormalFlow BoxModelNew PosR"]})
html_names = links[0].findAll("a")
citylist = []

for i in html_names:
    m = re.search('(href=\")(.*)(\" title)', str(i))
    citylist.append(m.group(2))
    print m.group(2)

#Define crawling function, k = city
def cityfunc(k):

    #Select City
    city = re.search('(used-cars\+in\+)(.*)', str(k)).group(2)

    #Figure out how many pages of results there are
    first_page = k + "/"
    soup2 = BeautifulSoup(urllib2.urlopen(first_page).read())
    t = soup2.html.head.title
    v = int(re.findall(r'\d+', str(t))[0])  #number of hits
    if (v%40) == 0:
        z = (v/40)                          #number of pages to scrape if remainder = 0
    else:
        z = ((v/40) + 1)                    #number of pages to scrape if otherwise

    #Open up blank spreadsheet
    fid = open("J:\\CASES\\India Car Prices\\Cardekho\\car_prices_" + city + ".csv","wb")
    writer = csv.writer(fid)

    #Open up blank master dictionary
    d = {}
    header = ["model", "fuel type", "price", "year", "KMS", "city", "seller"]
    d[000] = header

    #Define scraper, n = page number
    def scrape(n):
        #Navigate to website
        URL = "http://www.cardekho.com/used-cars+in+" + city + "/" + str(n)
        soup = BeautifulSoup(urllib2.urlopen(URL).read())

        #Begin parsing HTML
        tags_names = soup.findAll("li",attrs={"class":["Model withimgheight", "Model withoutimgheight"]})
        tags_price = soup.findAll("li",attrs={"class":["Price withimgheight", "Price withoutimgheight"]})
        tags_year = soup.findAll("li",attrs={"class":["Year withimgheight", "Year withoutimgheight"]})
        tags_kms = soup.findAll("li",attrs={"class":["Kms withimgheight", "Kms withoutimgheight"]})
        tags_city = soup.findAll("li",attrs={"class":["City withimgheight", "City withoutimgheight"]})
        tags_seller = soup.findAll("li",attrs={"class":["Seller HAuto"]})                                            
                      
        for x in range(len(tags_names)):
            #grabs the full html code fragment for the 'name' field (tag + text + /tag)
            html_names = tags_names[x].findAll("a")
            #grab the text in between the tags for 'name' (need if statement to correct for image/no image)
            if len(html_names) < 2:
                clean_name = str(''.join(html_names[0].findAll(text=True)))
            else:
                clean_name = str(''.join(html_names[1].findAll(text=True)))
            #grab the fuel type from the tags_names html chunk
            fuel_names = tags_names[x].findAll("span")
            #rest of the data fields can go into a small data dictionary
            data = {
            'fuel': str(''.join(fuel_names[0].findAll(text=True))),
            'price': str(''.join(tags_price[x].findAll(text=True))),
            'year': str(''.join(tags_year[x].findAll(text=True))),
            'kms': str(''.join(tags_kms[x].findAll(text=True))),
            'city': str(''.join(tags_city[x].findAll(text=True))),
            'seller': str(''.join(tags_seller[x].findAll(text=True)))
            }
            #dump all data fields into one row
            row = [clean_name, data['fuel'], data['price'], data['year'], data['kms'], data['city'], data['seller']]
            #create dictionary index
            index = (n*100) + x
            #dump row into master dictionary
            d[index] = row    

    #Loop function over z pages
    for a in range(1, (z+1)):
        scrape(a)

    #Write master dictionary to spreadsheet
    for key in sorted(d):
        writer.writerow(d[key])  
    fid.close()

#Loop over list of cities
for k in citylist:
    cityfunc(k)

