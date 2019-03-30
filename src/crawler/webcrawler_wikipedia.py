#Add Code
import scrapy

try: 
    from googlesearch import search 
except ImportError:  
    print("No module named 'google' found") 
  
# to search 
query = input() + " wiki";
strs = ["" for x in range(10)]
count=0;
for j in search(query, tld="com", num=10, stop=1, pause=2): 
    if count<10 :    
	strs[count]=j
    	count+=1
start_urls=[]
for st in range(5):
	print(strs[st])
class spider1(scrapy.Spider):
	name = 'Wikipedia'
 	start_urls = [strs[0],strs[1],strs[2],strs[3],strs[4]]
	def parse(self, response):
		string = ["" for x in range(5)]
		count=0       	
		for e in response.css('div#mw-content-text>div>p'):
			if count<5 :			
				string[count]= { 'para' : ''.join(e.css('::text').extract()).strip() }
				count+=1
		for i in range(5):		
			print(string[i])
