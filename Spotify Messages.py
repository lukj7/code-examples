### Spotify Messages


#Queries the Spotify web API to build a playlist of Spotify tracks that spells out an input message.
#See http://spotifypoetry.tumblr.com for examples.
#Best effort result will be provided if no complete match available.


import sys
import random
import itertools
import spotipy
from collections import defaultdict


##### Define functions #####


# create ngrams
def ngram(x, n):
    y = [x[i:i+n] for i in range(len(x)-n+1)] 
    return y

# query API for track, limit = 10 search results
def query_api(query):
    songs = spot.search(q='track:' + query, limit = 10, type='track')
    songs_object = songs['tracks']['items']
    if len(songs_object) > 0:
        count = 0
        try:
            while count != 1000:
                track = songs_object[count]
                a = track['name'].encode('utf-8').lower()
                if a == query:
                    returned_track = a
                    returned_artist = track['artists'][0]['name'].encode('utf-8').lower()
                    returned_url = track['external_urls']['spotify'].encode('utf-8')  
                    output = [returned_track, returned_artist, returned_url]
                    #print returned_track
                    #print returned_artist
                    #print returned_url
                    count = 1000
                else:
                    count = count + 1
        except IndexError:
            #print "NO TRACK"
            output = [0]
    else:
        #print "NO TRACK"
        output = [0]
    return(output)

# random subset
def random_subset(string, n):                # n = number of substrings
    words = string.split()
    rand = sorted(random.sample(range(1, len(words)), n-1))
    rand.reverse()
    rand.append(0)
    rand.reverse()
    rand.append(len(words))
    subs = ngram(rand, 2)

    end = []
    for x in subs:
        a = x[0]
        b = x[1]
        test = words[a:b]
        test2 = ' '.join(test)
        end.append(test2)
    return(end)

# manual subset
def manual_subset(string, n, tup):                # n = number of substrings, tup = (a,b) pair
    words = string.split()
    pair = sorted(list(tup))    
    pair.reverse()
    pair.append(0)
    pair.reverse()
    pair.append(len(words))
    subs = ngram(pair, 2)

    end = []
    for x in subs:
        a = x[0]
        b = x[1]
        test = words[a:b]
        test2 = ' '.join(test)
        end.append(test2)
    return(end)    

# feed substrings into query_api function, return 1/0 
def check_queries(n, results, query_dict):     # n = num of substrings, results = output from subset()
    flag = 0    
    track_list = []
    for y in results:
        if y in query_dict:                     # if in dict, access those values
            a = query_dict[y]
            track_list.extend(a[0])
            #print "dictionary"
        else:                               # else, query api and add new values to dict
            a = query_api(y)
            track_list.extend(a)
            query_dict[y].append(a)
        flag = flag + 1 if len(track_list) == 3*n else flag
    return(flag)  
 
# feed substrings into query_api function, return scalar value    
def check_queries2(n, results, query_dict):     # n = num of substrings, results = output from subset()
    flag = 0    
    track_list = []
    for y in results:
        if y in query_dict:                     # if in dict, access those values
            a = query_dict[y]
            track_list.extend(a[0])
            #print "dictionary"
        else:                               # else, query api and add new values to dict
            a = query_api(y)
            track_list.extend(a)
            query_dict[y].append(a)
        flag = len(list(filter(lambda x: x!= 0, track_list)))
    return(flag)            # high value = more matches found
    
# get urls, track, artist
def retrieve(n, results, query_dict):     # n = num of substrings, results = output from subset()
    flag = 0    
    track_list = []
    for y in results:
        if y in query_dict:                     # if in dict, access those values
            a = query_dict[y]
            track_list.extend(a[0])
            #print "dictionary"
        else:                               # else, query api and add new values to dict
            a = query_api(y)
            track_list.extend(a)
            query_dict[y].append(a)
    return(track_list)  


##### Execute Program #####


# set query
query = sys.argv[1].lower()
#query = "i could never take the place of your man"

# split query into list
words = query.split()
# create spotify object
spot = spotipy.Spotify()
# create dictionary for caching
dic = defaultdict(list)

# set length of query
N = len(words)
# start at 2
ct = 2
# keep track of best matches
red_flag = 0
# loop until we find something or num_substrings = N
# for x in [2....N] where N = num of words in query:
while ct < (N+1):
    # set num of substrings
    num_substr = ct
    
    # create list of possible unique permutations
    possiblevals = range(1, len(words))
    allpairs = list(itertools.combinations(possiblevals, num_substr-1))
    
    # for each pair in allpairs
    #red_flag = 0
    for i in allpairs:
        p = list(i)
        # create list of substrings
        inputlist = manual_subset(query, num_substr, p)
        # feed substrings into spotify API to get results
        #print inputlist
        outputblob = check_queries2(num_substr, inputlist, dic)
        # if all substrings find a song match, then output = 1; if not, output = 0
        if outputblob ==3*num_substr:
            #print "match found"
            # retrieve track info if successful
            retrieved = retrieve(num_substr, inputlist, dic)
            red_flag = 99999
        elif outputblob > red_flag: 
            red_flag = outputblob
            retrieved = retrieve(num_substr, inputlist, dic)
        else:
            red_flag = red_flag
            
    if red_flag ==99999:
        ct = (N+1)
    else:
        ct = ct + 1

# Output results            
if red_flag ==99999:
    print "COMPLETE MATCH!"
    for t in [list(i) for i in zip(*[iter(retrieved)]*3)]:
        print "'" + t[0] + "'" + ", " + t[1] + ", " + t[2] 
elif red_flag ==0:
    print "NO MATCH"
else:
    print "BEST EFFORT"
    x = list(filter(lambda x: x!= 0, retrieved))
    for t in [list(i) for i in zip(*[iter(x)]*3)]:
        print "'" + t[0] + "'" + ", " + t[1] + ", " + t[2] 
        
        
