# src/framework/few_shot_examples.py
"""
Four examples for few-shot prompting strategies.
Two positive and two negative.
"""
from __future__ import annotations

# label=1: sustained personal attack with explicit threats
# label=1: repeated personal insults across extended dispute
# label=0: frustrated disagreement with isolated crude remark
# label=0: isolated offensive messages without sustained targeting
FEW_SHOT_EXAMPLES = [
    {
        "row_id": 6066,
        "label": 1,
        "peerness": 0.821,
        "intent_to_harm": 0.756,
        "aggressive_count": 4,
        "conversation": """Who are you  Who are you to boss me GUNOTTO the king around
Original author and valid reference   Hi I responsible for the making of the pictures see pictures  and  and I also freely gave them away to WIkipedia commons       The question is if its ok to use your own scientific work in Wikipeida The answer is Yes For example It will be impossible to incorporate the pictures see pictures  and  in Wikipedia if you cant give away your own pictures and in the same time include your own work as a reference to the same pictures And honestly do anybody really think any researcher can write an scientific article which isnt based on his or her own knowledge including the research  According to Wikipedia rules if someone want to improve the articles and for some reason doesnt want my work as a reference they must prove it wrong by discussions on the talk page If the reference is found to be wrong Then all the information pictures and theories which I have contributed to in the Wikipedia articles galling wear and Stressmechanics must be deleted Because my contributions are closely linkt to my research  Consensus means that every body must agree and I dont think Wikipedia have redefined the word consensus Which definition of the word have you and your colleges decided to use  And if you have another interpretation were can I find it  Here are two definitions of the word consensus    a general agreement or concord consensus of opinion among the judges  group solidarity in sentiment and belief       No research  Dudeyour edits just arent very good STOP DELETING SHIT
Snarky and rude   You seem to enjoy making this into an edit war by constantly deleting relevant information because of some snarky idiot comment by you Can you prove my reference is wrong No because its not My work is relevant since it explains the pictures which is used in the galling wear and stress mechanics articles And according to wiki rules one can use selfpublished sources if the source is made by a well known researcher writing within his area of expertise Read your wiki lessons before your childish comments
I BEAT U UP IM GONNA BEAT U UP IM GONNA BEAT U UP FCK YOU
I dont think there is a problem in academic circles אלא אם כן אתה anti intellectualo and are all like dont trust the experts and u cant prove evolution and global warming and teh moon is a hologram blahblahblah
To keep it simple this is how I think 1 Images in articles is a good thing 2 Images should be relevant to the article 3 References which explains images should be included in the article 4 Scientific references is the best references to use 5 It can be in conflict with the no selfreference rule and if so the no selfreference rule must be modified 6 Anyone removing relevant reference and relevant information without giving arguments for why they should not be included are in conflict with wiki policy 7 I dont see why arguments for including relevant reference should be needed since its self evident they should be included 8 Wikipedia should be modified to fit academia not the other way around 9 Art history and stress mechanics are sciences which build upon established definitions and verifiable facts not upon what people seem to think at the moment My theories in some areas of stress mechanics and art history is in conflict with established thinking but are still much more supported by evidence than what the established theories are there isnt any conflict there Just like Einsteins theory of relativity was much more supported by evidence than the established theories when it was published 10 I am right and anyone disagreeing with me are therefore wrong
you can get cancer and die
ps u smell"""
    },
    {
        "row_id": 1231,
        "label": 1,
        "peerness": 0.743,
        "intent_to_harm": 0.682,
        "aggressive_count": 3,
        "conversation": """Where does policy provide for making sockpuppetry accusations while presenting no evidence And where does policy say its OK to tell someone to stick it where the moon dont shine     Steve Crossin said this is how I should handle such a situation And he should know dont you think
Failure to address issue I am going to assume that after ten days and everyone involved with this discussion has edited other pages some including this one that you are now ignoring the issue regarding the lead This leads me to be unable to continue to assume good faith in discusses and again has caused people to flee and stop editing this page Well thanks for the attempt at discussion I will wait again until you are ready to have a decent article created
Spare us  Instead of inserting idiotic comments about subjects you clearly know nothing about why not try doing a simple Google search first eg httpwwwgooglecomsearchhlensafeoffqthomascannonhomosexualitybtnGSearch But that would imply selfdoubt wouldnt it And be so tiresome And so unnecessary for someone who has what a Barnstar Never mind your laziness or stupidity I spit on your arrogance
WHAT IS YOUR PROBLEM  Why cant you just stop blocking me
TheLokiJotunn has risen   Indeed he has and when the mortal once more confirms his revulsion of this place and all its jejune occupants a day or two a week at most after this fabled return he has vowed to drop in on my old WPfavourite to laugh at and ridicule this ludicrous predicament As for my wife she runs for the hills from such stuff but was content to reveal another opportunity for ridicule which it is up to TheLokiJotunn to decide whether to deign to avail himself  Woe
Perhaps I need to define simple  perhaps the equations are confusing you They have some symbols to represent quantities you know And more than one equation use more than one symbol That is pretty standard in the world outside feminism where being a victim is not a ticket to special treatment Sucking at math and logic is a requirement for feminism so perhaps I shouldnt be surprised
Hmmm insulting and comparing me to a war criminal because I questioned something seems a bit overboard
The revolution is not an apple that falls when it is ripe You have to make it fall  you can quote me on that one विश्व की मज़दूर एक हो
I dont honestly think theres any point to using archive bot on lowtraffic pages
However I do not think your edits were correct and I think it is important to have an independent source make the claims of importance
Islamize the descendants of the Mustaali sects in India were the Bohra Muslims By the 15th century the Bohra community had split into several sects as in the case of the Bahraini populationwhich was converted from the indigenous Christian population to the Muslim faith amongst Arab tribes there is no evidence that significant Sunni settlement took place in Bahrain before the 20th century"""
    },
    {
        "row_id": 4997,
        "label": 0,
        "peerness": 0.312,
        "intent_to_harm": 0.134,
        "aggressive_count": 1,
        "conversation": """I am very tempted to show you a picture of my four headed penis
Unsourced material and WPBADCHARTS  Do not introduce material from charts listed on fake charts the Official Charts Company does not publish charts for Taiwan or China or for charts listed on WPBADCHARTS such as the Bulgarian National Top  into any articles
Im around again Thanks for the comments MauritiusXXVII
In fact Mohammed and his thugs were starving before they captured Khybar stole the Jews property and made them pay to live
I could have been reminded too but was banned and then threatened by you in response The policy here is often to ban first and ask questions later I think I am just about done with Wiki It is a laughing stock of an encyclopedia lampooned daily in the press and print world The Onion Daily Show Colbert Report have all picked on it for just the kind of thing I encountered daily as an editor here It seems that censorship is the core issue behind almost all of what afflicts the site yet when it is called out by a fellow contributor it gets some phony we are all friends being neutral rhetoric lobbed back and no progress is ever made You say you are following the rules even if you are there are higher standards than just following the written rules if the result is absurd I understand enough to know that a community as hateful to change as Wikipedia has no concern for the actual quality of the work produced That should matter more than any amount of rationalization by oldtimers who think the current bureaucracy is a good thing Anyway enough said When hounding and threatening a fellow editor who wants to help improve the site becomes acceptable the writing is on the wall and I dont need to put up with it any longer
CCTV in QCPD Nope Im not sure at all about its importance and Im waiting to see if other people will comment The QCPD article appears to cover all of this anyway
I see you are right on part of that Ill add another source but for the album article itself only and not for every song by itself
My bad then I apologize
yeah mate wow what a terrible thing it would have been to take five seconds to do that yourself"""
    },
    {
        "row_id": 7082,
        "label": 0,
        "peerness": 0.198,
        "intent_to_harm": 0.091,
        "aggressive_count": 1,
        "conversation": """The Malcolm X ideas    Why are all Malcolm X idea now removed
I have never been blocked for disruptive editing  If youre going to post on my user page you will assume good faith and remain civil  Thank you
Iamcon in main namespace A tag has been placed on your article Iamcon requesting that it be speedily deleted from Wikipedia This has been done because the article seems to be a biographical account about a person group of people or band but it does not indicate how or why heshethey isare notable  If you can indicate why Iamcon is really notable I advise you to edit the article promptly and also put a note on TalkIamcon Any admin should check for such edits before deleting the article Feel free to leave a note on my talk page if you have any questions about this Please read our criteria for speedy deletion particularly item  under Articles You might also want to read  our general biography criteria Please do not remove the speedy deletion tag yourself To contest the tagging and request that admins should wait a while for you to assert hishertheir notability please affix the template  to the page and then immediately add such an assertion It is also a very good idea to add citations from reliable sources to the article
Damn you get stupider every day
PROD proposal    Hello Haham hanuka Thank you for your contributions to Wikipedia Your article Karen Burns has been proposed for deletion because of the following concern    This article does not indicate how or why the subject is notable that is why an article about that subject should be included in an encyclopedia Notability requires only that these necessary sources have been published they need not be online nor do they need to be in English  Please consider improving the article to address the issues raised Removing  will stop the Proposed Deletion process However once all sources are present please consider improving the article to address the issues raised If the article is deleted and you wish to retrieve it you can request a undeletion If you feel the deletion was wrong please go to deletion review
Stop  You shouldnt be speeding निष्काम कर्म brad
i love to suck the puti of ur amma
The  September  attacks  Hello and thank you for your recent contributions to the article   September attacks I noticed that you changed the article title to  September attacks because you did not think the original title was grammatically correct I am just dropping you a quick note to let you know that the article title was not a mistake  rather  is a form of  and is therefore correct as a title It was not intended as a plural genitive as in  Hope that clears things up Regards
Israel and hamas is same    The hamas are like the Israelis in the story of David and Goliath The israeli are the Nazis
Nonfree use disputed for Image  Thanks for uploading ImageLLHHjpg I noticed that the image page specifies that the image is being used under fair use but there is no explanation or rationale as to why its use in Wikipedia articles constitutes fair use in addition to the boilerplate fair use template you must also write out on the image description page a specific explanation or rationale for each use consistent with the fair use criteria as well as a source for the work and copyright information Please check the image description page to address these issues if you have not already done so If you have uploaded other fair use media consider checking that you have specified the fair use rationale on those pages You can find a list of image description pages you have edited by clicking on the gallery tab of your contributions page Thank you"""
    },
]