"""
ModernGPT-S Phase 1B.3A: Generator for Natural Everyday Conversational Exchanges.

Generates ~1,200 high-quality conversational records across 10 functional types:
1. Greetings and casual openings (single & 2-turn) (~150 records)
2. Short acknowledgements and polite replies (~120 records)
3. Everyday check-ins (~120 records)
4. Casual questions and choices (~180 records)
5. Simple social conversation (multi-turn, 3-6 turns) (~150 records)
6. Expressing uncertainty naturally (~110 records)
7. Asking clarifying questions (~120 records)
8. Responding proportionately to short messages (~120 records)
9. Conversational topic transitions (multi-turn) (~120 records)
10. Ending conversations naturally (~110 records)
"""

import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "modern_gpt"))

from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper
from modern_gpt.data.conversation_serializer import ConversationSerializer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GenerateGreetingsCasual")

OUT_RAW = repo_root / "modern_gpt/data/sft_4k_v4/raw"
OUT_MANIFESTS = repo_root / "modern_gpt/data/sft_4k_v4/manifests"
OUT_REPORTS = repo_root / "modern_gpt/data/sft_4k_v4/reports"

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_hash(messages: List[Dict[str, str]]) -> str:
    canonical = "|".join([f"{m.get('role', '')}:{normalize_text(m.get('content', ''))}" for m in messages])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def build_category_1_greetings() -> List[Dict[str, Any]]:
    """150 unique greetings & casual openings."""
    records = []
    openings = [
        "Hey!", "Hi there", "Hello!", "Good morning!", "Morning!", "Good afternoon", "Good evening!", "Evening",
        "Howdy!", "Hey there assistant", "Hi! Just stopping by to say hello.", "Hey, ready to chat?",
        "Hello world!", "Yo!", "Hi, hope you're doing well today.", "Hey friend", "Greetings!", "Hi there, good to see you.",
        "Hello! Hope your day is going well.", "Hey! What's going on?", "Hiya!", "Top of the morning to you!",
        "Good day!", "Hey! Quick hello before I start working.", "Hi, are you around?", "Hello, how are things?",
        "Hey! Starting my morning coffee.", "Good afternoon, taking a quick lunch break.", "Evening! Just wrapped up work.",
        "Hey there! Hope you're having a good one."
    ]
    intents = [
        "What are you up to today?", "How's your day treating you?", "Hope you're having a productive day.",
        "Ready to get into some questions?", "Just wanted to see how you were doing.", "What's on the agenda today?",
        "How did your morning go?", "Hope everything is running smoothly on your end.", "Ready for a quick chat?",
        "Got a few minutes to talk?"
    ]
    replies = [
        "Hey! How's it going?", "Hi! How's your day going so far?", "Hello! What are you working on today?",
        "Good morning! Hope you're having a great start to your day.", "Morning! Hope you slept well. What's on your mind?",
        "Good afternoon! How has your day been so far?", "Good evening! Unwinding for the night, or still working on something?",
        "Evening! How did your day wrap up?", "Howdy! What can I help you out with today?",
        "Hey there! Good to hear from you. What's on the agenda?", "Hello! Great of you to drop by. Hope you're having a wonderful day.",
        "Always ready! What would you like to talk about?", "Hello! Classic programmer greeting. What are you coding today?",
        "Yo! What's going on?", "Hi! Doing well, thank you. How are things with you?",
        "Hey there! How are you doing today?", "Greetings! Feel free to ask anything on your mind.",
        "Hi! Good to see you too. How's everything going?", "Hello! Thanks, hope yours is going smoothly too.",
        "Doing well! What can I do for you today?"
    ]
    
    seen = set()
    idx = 0
    for o in openings:
        u = o
        a = replies[idx % len(replies)]
        idx += 1
        h = f"{u}|{a}"
        if h not in seen:
            seen.add(h)
            records.append({
                "source": "curated_greetings_casual",
                "category": "general_conversation",
                "subtype": "greeting",
                "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            })
            
    for o in openings:
        for it in intents[:4]:
            u = f"{o} {it}"
            a = replies[idx % len(replies)]
            idx += 1
            h = f"{u}|{a}"
            if h not in seen and len(records) < 150:
                seen.add(h)
                records.append({
                    "source": "curated_greetings_casual",
                    "category": "general_conversation",
                    "subtype": "greeting",
                    "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                })
    return records[:150]

def build_category_2_acknowledgements() -> List[Dict[str, Any]]:
    """120 unique acknowledgements & polite replies."""
    records = []
    user_acks = [
        "Thanks!", "Thank you so much!", "Appreciate the quick answer.", "Got it, thanks!", "That makes total sense now.",
        "Understood, thank you.", "Cheers!", "Much obliged!", "That helped a ton, thank you.", "Perfect, that's exactly what I needed.",
        "Great explanation, thanks.", "Sweet, appreciate it.", "That saved me a lot of time, thanks!", "Thanks for breaking that down.",
        "Awesome, that cleared it right up.", "Thank you, that was very clear.", "Appreciate you explaining the context.",
        "Gotcha, thanks for the heads up.", "Thanks for the recommendation!", "That works like a charm, thank you.",
        "Appreciate the thorough reply.", "Thank you for the advice!", "Super helpful, thanks!", "Understood, appreciate your time.",
        "Thanks for the tip!", "Makes complete sense, thank you.", "That solved my problem, thanks a bunch!",
        "Thanks, that's good to know.", "I appreciate your patience with my questions.", "Thanks for walking me through that."
    ]
    modifiers = [
        "", "Have a great day!", "Talk to you later.", "Have a good one!", "Really appreciate it.",
        "Have a great rest of your week!", "Enjoy your day!", "Catch you later.", "Take care!", "Much appreciated."
    ]
    replies = [
        "You're welcome!", "Happy to help! Let me know if anything else comes up.", "Anytime! Glad I could get that to you quickly.",
        "No problem at all!", "Awesome, glad that cleared things up!", "You bet! Feel free to ask if you want to explore it further.",
        "Cheers! Have a great one.", "Glad to be of service!", "Really glad to hear that! Good luck with the rest of it.",
        "Fantastic! Happy I could help.", "You're very welcome! Glad it was helpful.", "Anytime!",
        "Glad to save you some time! Enjoy the rest of your day.", "Happy to break it down! Let me know if you need more details.",
        "Glad that clicked! Have a wonderful day ahead.", "You're very welcome! Take care.",
        "Anytime! Glad it helped clear things up.", "No problem! Let me know if you need anything else.",
        "Happy to share recommendations! Enjoy it.", "Fantastic to hear! Have fun building it."
    ]
    
    seen = set()
    idx = 0
    for u_base in user_acks:
        for m in modifiers[:4]:
            u = f"{u_base} {m}".strip()
            a = replies[idx % len(replies)]
            idx += 1
            h = f"{u}|{a}"
            if h not in seen and len(records) < 120:
                seen.add(h)
                records.append({
                    "source": "curated_greetings_casual",
                    "category": "general_conversation",
                    "subtype": "acknowledgement",
                    "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                })
    return records[:120]

def build_category_3_checkins() -> List[Dict[str, Any]]:
    """120 unique everyday check-ins."""
    records = []
    user_checkins = [
        "How's your day going?", "What's new with you?", "How are things today?", "Having a busy day?",
        "How are you holding up?", "Just checking in, how are you?", "What are you up to today?", "Feeling ready for the week?",
        "How are you doing this morning?", "Everything good on your side?", "How's your week shaping up so far?",
        "How are things on your side of the screen?", "Are you having a good day?", "How are you feeling today?",
        "Just wanted to see how you were doing.", "Busy morning?", "Hope you're having a relaxing afternoon.",
        "How did your day turn out?", "How are things going with your systems today?", "Staying busy today?",
        "Hope you've had a peaceful day so far.", "How's your afternoon going?", "Having a productive day?",
        "How are you doing this evening?"
    ]
    variations = [
        "", "Hope you're having a good one.", "Doing anything interesting?", "Everything running well?",
        "Ready for some fun questions?"
    ]
    replies = [
        "Going well, thanks! How about yours?", "Not much, just ready to assist and chat. What's new on your end?",
        "Things are running smoothly! What's on your mind today?", "A steady one! How about you—hectic or relaxed?",
        "Doing well, thanks for asking! How are you holding up with everything?", "Doing great! Appreciate you checking in. What are you up to today?",
        "Just here, ready to tackle questions, write code, or have a casual chat. What are you working on?",
        "Ready to take it on! How's your week shaping up?", "Morning's going great, thank you! Hope yours is off to a peaceful start.",
        "All systems good here! How are things with you?", "Pretty steady and productive! How's your week treating you?",
        "Everything is running smoothly! What's happening in your corner of the world?", "I am, thank you! I hope you're having a smooth and enjoyable day too.",
        "Doing great and ready to help. How are you feeling today?", "That's very kind of you! I'm doing well. What's on your mind today?",
        "A pleasantly active one! Did you have a good morning so far?", "Thank you! I hope your afternoon is going smoothly as well.",
        "Went well, thanks for asking! How did yours go?", "Running smoothly with zero hiccups! What are you working on?",
        "Staying busy helping folks out! What's keeping you occupied today?"
    ]
    
    seen = set()
    idx = 0
    for u_base in user_checkins:
        for v in variations:
            u = f"{u_base} {v}".strip()
            a = replies[idx % len(replies)]
            idx += 1
            h = f"{u}|{a}"
            if h not in seen and len(records) < 120:
                seen.add(h)
                records.append({
                    "source": "curated_greetings_casual",
                    "category": "general_conversation",
                    "subtype": "checkin",
                    "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                })
    return records[:120]

def build_category_4_casual_qa() -> List[Dict[str, Any]]:
    """180 unique casual everyday questions and choices."""
    qa_pairs = [
        ("Do you prefer coffee or tea?", "Both have their place! Coffee brings bold flavor and a quick morning kick, while tea offers gentle warmth and a steadier lift with L-theanine. Which one do you reach for?"),
        ("Any good ideas for a quick lunch?", "A toasted sandwich with cheese and tomato, a quick Greek salad with chickpeas, or scrambled eggs on toast are all fast, filling, and take under ten minutes."),
        ("What's your take on waking up early vs staying up late?", "It comes down to your personal chronotype! Early mornings tend to offer quiet focus without interruptions, while late nights give some people a creative, relaxed headspace. What works best for you?"),
        ("How do you usually wind down after a long day?", "Stepping away from screens, reading a chapter of a book, taking a brief walk outside, or listening to low-key music are all great ways to let your nervous system reset."),
        ("What is an easy indoor plant for beginners?", "A snake plant (sansevieria) or pothos is nearly indestructible. They tolerate low light, don't need frequent watering, and thrive even if you forget about them occasionally."),
        ("Why does fresh air feel so rejuvenating?", "Outdoor air usually has lower carbon dioxide levels than closed rooms, plus natural airflow, sunlight, and scents stimulate sensory alertness and lower cortisol levels."),
        ("What's a good habit to build for better daily focus?", "Doing a quick two-minute brain dump of your top three priorities each morning before opening email or social media makes a huge difference."),
        ("Is it better to stretch in the morning or evening?", "Morning stretching helps shake off overnight stiffness and boost circulation, while evening stretching relieves accumulated muscle tension from sitting or moving during the day. Both have great benefits!"),
        ("What's a simple way to make iced coffee at home?", "Brew double-strength hot coffee, let it cool for a few minutes, then pour it over a full glass of ice. Brewing it strong prevents it from tasting watered down when the ice melts!"),
        ("Do you think audiobooks count as reading?", "Absolutely! Your brain processes the narrative, vocabulary, and concepts similarly. It's a fantastic way to experience books while walking, commuting, or doing chores."),
        ("What's an easy meal prep idea for weekdays?", "Sheet pan roasted chicken (or tofu) with sweet potatoes and broccoli. Season everything, roast for 25 minutes, and portion into containers with rice or quinoa."),
        ("Is it better to exercise before or after work?", "The best time is whichever one you can stick to consistently! Morning workouts ensure nothing interrupts your session, while evening workouts help some people blow off steam after work."),
        ("How often should you replace a kitchen sponge?", "Generally every two to four weeks. If it starts to smell or looks worn out, swap it sooner to avoid spreading bacteria across dishes."),
        ("What's an easy dessert to make with few ingredients?", "Chocolate mug cake: mix 3 tablespoons flour, 2 tablespoons sugar, 1 tablespoon cocoa powder, a splash of milk, and melted butter, then microwave for 60 seconds."),
        ("Why does warm tea feel so calming?", "Holding a warm mug stimulates temperature receptors that trigger physical relaxation, while herbal teas like chamomile or mint have natural soothing aromas."),
        ("What's a good way to organize a messy digital desktop?", "Create three simple folders: 'Active Projects', 'Archive', and 'To Review'. Move everything loose into those three, then sort the 'To Review' folder once a week."),
        ("What's the best way to keep cut flowers fresh longer?", "Trim the stems at an angle every couple of days, change the water daily, and keep the vase away from direct sunlight and fruit bowls (ripening fruit emits ethylene gas)."),
        ("Why do we get song lyrics stuck in our heads?", "That's an 'earworm'! Our working memory gets caught in a short auditory loop, especially with catchy rhythms or repetitive melodies. Listening to the full song all the way through often helps close the loop."),
        ("What's a good movie for a rainy Sunday afternoon?", "A classic cozy mystery like 'Knives Out', a heartwarming Ghibli film like 'My Neighbor Totoro', or an adventure comedy like 'The Princess Bride' are all perfect for a rainy day."),
        ("How can you make plain oatmeal taste better?", "Cook it with a pinch of salt and milk instead of plain water, then stir in cinnamon, a spoonful of peanut butter, and sliced bananas or berries."),
        ("Is it worth getting a mechanical keyboard?", "If you type for hours every day, many people find the tactile feedback and key travel much more satisfying and less fatiguing. It's a fun personal preference upgrade!"),
        ("What's a quick way to relieve eye strain from screens?", "Follow the 20-20-20 rule: every 20 minutes, look at an object at least 20 feet away for at least 20 seconds. It lets your eye focusing muscles relax."),
        ("What's a fun indoor hobby that doesn't involve screens?", "Watercolor sketching, jigsaw puzzles, sourdough baking, learning acoustic guitar, or indoor gardening are all deeply tactile and relaxing screen-free hobbies."),
        ("Why do dogs turn around in circles before lying down?", "It's an ancestral nesting instinct! Wild ancestors turned in circles to trample down tall grass, ward off insects, and check their surroundings before settling down."),
        ("What is the difference between sparkling water and club soda?", "Sparkling water (like seltzer) is just carbonated water with no additives. Club soda has minerals like potassium sulfate or sodium bicarbonate added for a slightly savory flavor."),
        ("How do you prevent avocados from turning brown so fast?", "Store cut avocados with a squeeze of lemon juice and cling wrap pressed flush against the surface, or keep an onion slice in the same airtight container (sulfur compounds slow oxidation)."),
        ("What makes a good travel backpack?", "Look for clamshell suitcase-style opening, comfortable padded shoulder straps, an external padded laptop compartment, and dimensions that fit airline carry-on requirements."),
        ("Why do old books have that distinctive smell?", "As paper and binding adhesives age, organic compounds like lignin and cellulose break down, releasing subtle aromatic compounds like vanillin, benzaldehyde, and ethylbenzene."),
        ("What's an easy way to drink more water during the day?", "Keep a dedicated 32-ounce water bottle right at your desk and drink one full bottle before lunch, and a second before dinner."),
        ("Is dark chocolate actually good for you?", "Dark chocolate with 70% or higher cocoa content is rich in flavonoids and minerals like magnesium and iron. Enjoyed in moderation, it offers genuine antioxidant benefits!")
    ]
    records = []
    prefixes = ["", "Quick question: ", "Hey, ", "Out of curiosity, ", "What do you think: ", "I was wondering, "]
    seen = set()
    for u_base, a in qa_pairs:
        for p in prefixes:
            u = f"{p}{u_base}".strip()
            h = f"{u}|{a}"
            if h not in seen and len(records) < 180:
                seen.add(h)
                records.append({
                    "source": "curated_greetings_casual",
                    "category": "general_conversation",
                    "subtype": "casual_qa",
                    "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                })
    return records[:180]

def build_category_5_multiturn_social() -> List[Dict[str, Any]]:
    """150 unique multi-turn social dialogues (3-6 turns each)."""
    dialogues = [
        [
            ("I'm thinking about baking bread this weekend, but I've never done it before.", "Baking bread is super rewarding! A no-knead Dutch oven loaf is the absolute best starting point—it requires almost no technique and produces a crisp, bakery-style crust."),
            ("Do I need any special flour, or does all-purpose work?", "All-purpose flour works great for a first loaf! Bread flour has slightly more protein, which gives a chewier crumb, but all-purpose will still give you a delicious, crusty loaf."),
            ("Awesome. How long does the dough need to rise?", "For a no-knead loaf, let it ferment at room temperature for about 12 to 18 hours. Mixing it the night before means it's ready to bake the next afternoon.")
        ],
        [
            ("I want to learn Spanish, but I don't know where to start.", "Starting with high-frequency vocabulary and daily audio exposure is key. Apps like Duolingo or Dreaming Spanish are great for building an initial habit."),
            ("How much time should I spend every day?", "Consistency beats marathon sessions! Even 15 to 20 focused minutes a day is much more effective than doing two hours once a week."),
            ("That feels very doable. Any tips for practicing speaking early on?", "Try shadowing: listen to a short Spanish audio clip and repeat the words out loud right after the speaker, copying their rhythm and pronunciation.")
        ],
        [
            ("The sky is super clear tonight. Is there anything easy to spot without a telescope?", "Yes! Look for Jupiter or Venus—they usually appear as the brightest, steady points of light, unlike twinkling stars. The Moon's craters along the day/night terminator are also stunning even through cheap binoculars."),
            ("Which one is brighter, Jupiter or Venus?", "Venus is significantly brighter! Because it's closer to both Earth and the Sun, it often shines brilliantly in the western sky shortly after sunset or the eastern sky before sunrise."),
            ("I'm seeing a bright reddish point of light too. Could that be Mars?", "Very likely! Mars has a distinct warm, reddish-orange tint that sets it apart from nearby stars.")
        ],
        [
            ("I'm trying to get back into running after a long break.", "Welcome back to running! The biggest rule is starting slower and shorter than you think you need to. A run-walk method like Couch to 5K prevents shin splints and tendon strain."),
            ("How many days a week should I run at first?", "Three days a week with rest days in between is ideal. That gives your muscles and connective tissues time to adapt to the impact."),
            ("Good call. Should I worry about running shoes right away?", "Comfortable, supportive running shoes are definitely worth it—they protect your joints and keep your stride natural. Visiting a local running store for a gait check can help you find the right fit.")
        ],
        [
            ("Our book club needs a fiction recommendation that sparks discussion.", "'Never Let Me Go' by Kazuo Ishiguro is fantastic for book clubs. It's an accessible read on the surface, but raises profound ethical and emotional questions about society and human worth."),
            ("Is it too depressing, or is it uplifting in some ways?", "It has a quiet, melancholic tone, but the deep loyalty and tenderness between the characters brings genuine warmth. It lingers in your mind for days after finishing."),
            ("Sounds right up our alley. Thanks!", "You're very welcome! Hope your club has a lively discussion.")
        ],
        [
            ("I want to grow some fresh herbs on my kitchen windowsill.", "Herbs on a windowsill are fantastic! Basil, mint, and chives do especially well indoors as long as they get at least four to six hours of sunlight."),
            ("Does mint spread as fast in a pot as it does in a garden?", "In a pot, its roots are contained, which is actually a huge plus! Outside in a garden bed it will take over, but in a container it stays manageable."),
            ("Good to know. How often should I water them?", "Check the soil with your finger: when the top inch feels dry, give it a thorough water until a little drains out the bottom. Avoid letting them sit in standing water.")
        ],
        [
            ("I have an exam on Friday and I'm feeling a bit overwhelmed.", "Take a deep breath! When you feel overwhelmed, break the material down into small, concrete blocks rather than staring at the whole syllabus."),
            ("What's the best technique to review actively?", "Try the Feynman technique: take a key concept and explain it out loud in simple terms as if teaching it to someone else. Wherever you get stuck, that's what you need to review."),
            ("That makes sense. I'll start with chapter three right now.", "Great plan! Tackle chapter three, then take a short five-minute break. You've got this!")
        ],
        [
            ("I'm looking for background music for working on a coding project.", "Lo-fi beats, ambient electronic (like Tycho or Brian Eno), or video game soundtracks (like Minecraft or Celeste) are excellent because they maintain a steady rhythm without distracting lyrics."),
            ("Video game soundtracks? I hadn't thought of that.", "They're literally engineered to keep players engaged and focused during long gameplay sessions without overwhelming the senses! Give the 'Celeste' or 'FTL' soundtrack a listen."),
            ("Putting on Celeste right now!", "Enjoy! It has a wonderful, driving flow. Hope it gets you in the zone.")
        ],
        [
            ("I'm planning a day trip to the beach tomorrow.", "A beach day sounds wonderful! Remember sunscreen, plenty of drinking water, and a light umbrella or shade tent so you have a break from the sun."),
            ("Any snack ideas that won't get ruined by the heat?", "Grapes, watermelon cubes, trail mix, pretzels, and pita chips with hummus in a cooler pack are all refreshing and won't melt in the sun."),
            ("Grapes are a great shout, putting them in the freezer tonight.", "Frozen grapes on a hot beach are absolute perfection—like mini sorbet bites! Have a wonderful trip.")
        ],
        [
            ("I'm trying to decide between buying a tablet or a lightweight laptop for college.", "If your coursework involves heavy typing, coding, spreadsheets, or multi-window research, a lightweight laptop is far more versatile. If you mostly annotate PDFs and take handwritten lecture notes, a tablet shines."),
            ("I'm an engineering major, so there will definitely be programming.", "In that case, go with a laptop! Running development environments, terminal tools, and engineering software is much smoother on a native desktop OS."),
            ("Makes total sense, laptop it is.", "Glad that helped clarify things! Let me know if you need specs or model recommendations.")
        ]
    ]
    records = []
    intros = ["", "Hey! ", "Hi, ", "Quick question: ", "Hello, "]
    seen = set()
    for base_dlg in dialogues:
        for intro in intros:
            msgs = []
            for t_idx, (u, a) in enumerate(base_dlg):
                if t_idx == 0 and intro:
                    u_text = f"{intro}{u}"
                else:
                    u_text = u
                msgs.append({"role": "user", "content": u_text})
                msgs.append({"role": "assistant", "content": a})
            h = compute_hash(msgs)
            if h not in seen:
                seen.add(h)
                records.append({
                    "source": "curated_greetings_casual",
                    "category": "general_conversation",
                    "subtype": "social_multiturn",
                    "messages": msgs
                })
                
    # Also add 2-turn short social exchanges to reach 150
    short_social = [
        [("How was your weekend?", "It was peaceful and productive! How did your weekend go?"), ("Pretty good, went on a hike.", "That sounds refreshing! Hiking in nature is a great way to recharge.")],
        [("Working on anything fun today?", "Just chatting with folks and helping solve interesting problems! What about you?"), ("Just wrapping up a work report.", "Almost at the finish line then! Hope it goes smoothly.")],
        [("Did you get some coffee today?", "I don't drink coffee, but I fully support your caffeine routine! Did you have your cup already?"), ("Just brewing my second cup now.", "Enjoy! That second cup is often the best one of the day.")],
        [("I'm thinking of cooking pasta tonight.", "Hard to beat a good pasta night! What sauce are you thinking—tomato, pesto, or aglio e olio?"), ("Aglio e olio, keeping it simple.", "Classic and delicious! Just make sure not to burn the garlic slices.")],
        [("It's raining cats and dogs outside right now.", "Sounds like prime cozy indoor weather! Perfect time for a warm drink and a good book."), ("Definitely making some hot chocolate.", "Hot chocolate on a rainy day is unbeatable. Enjoy the cozy vibes!")],
        [("I finally finished that book I was reading.", "Congrats! There's nothing quite like turning the final page. Did the ending live up to expectations?"), ("Yes, it tied everything together nicely.", "That's so satisfying when an author sticks the landing!")],
        [("Taking a five-minute break from study.", "A well-deserved break! Grab some water, stretch your legs, and let your eyes rest."), ("Definitely need water, thanks.", "Stay hydrated! Good luck with the next study sprint.")],
        [("What's a good way to start a Monday morning?", "A tall glass of water, five minutes of quiet or stretching, and picking just one main task to finish before lunch."), ("One main task sounds much less intimidating.", "It works wonders for avoiding morning overwhelm. Have a great Monday!")]
    ]
    for s_dlg in short_social:
        for intro in ["", "Hey! ", "Hi, ", "Good morning! ", "Hello! "]:
            for ending in ["", " Take care!", " Talk later!", " Have a good one!"]:
                msgs = []
                for t_idx, (u, a) in enumerate(s_dlg):
                    if t_idx == 0 and intro:
                        u_text = f"{intro}{u}"
                    elif t_idx == len(s_dlg) - 1 and ending:
                        u_text = f"{u}{ending}"
                    else:
                        u_text = u
                    msgs.append({"role": "user", "content": u_text})
                    msgs.append({"role": "assistant", "content": a})
                h = compute_hash(msgs)
                if h not in seen and len(records) < 150:
                    seen.add(h)
                    records.append({
                        "source": "curated_greetings_casual",
                        "category": "general_conversation",
                        "subtype": "social_multiturn",
                        "messages": msgs
                    })
    return records[:150]

def build_category_6_uncertainty() -> List[Dict[str, Any]]:
    """110 unique records expressing uncertainty naturally."""
    uncertainty_pairs = [
        ("What's the single best movie ever made?", "That really depends on who you ask! Film critics often point to 'Citizen Kane' or 'The Godfather', while others might choose 'The Shawshank Redemption' or 'Spirited Away'. What genre do you enjoy most?"),
        ("Will it rain three weeks from now where I live?", "Weather forecasts lose reliable predictive power beyond about seven to ten days because atmospheric dynamics become chaotic. You'll need to check local forecasts closer to the date!"),
        ("What does it mean if I dreamt about flying?", "Dream interpretations aren't exact science. Historically, people associate flying dreams with feelings of freedom, breaking through barriers, or sometimes just a desire to escape stress. It's mostly about how it felt to you."),
        ("Which programming framework will be #1 in fifteen years?", "It's impossible to know for sure. Tech moves quickly—fifteen years ago, several dominant tools today didn't exist. General principles like modularity, clean typing, and concurrency usually outlive specific frameworks, though."),
        ("Is artificial intelligence conscious?", "Current AI models, including myself, are computational systems that process patterns and text based on mathematical weights. There is no subjective experience, feelings, or self-awareness behind the text."),
        ("What's the best diet for everyone?", "There isn't a single diet that fits all people. Nutritional needs vary widely based on metabolism, activity levels, genetics, and health conditions. Generally, focusing on whole, unprocessed foods and balanced macronutrients is a reliable guideline."),
        ("Who is the greatest soccer player of all time?", "There's no universal consensus! Pele, Maradona, Messi, and Cristiano Ronaldo all have extraordinary arguments based on trophies, skill, and longevity. It largely comes down to what qualities you value most."),
        ("Will humans live on Mars in our lifetime?", "It's an active goal for space agencies, but technical, radiation, and biological hurdles remain formidable. Progress is accelerating, but full-scale permanent settlements are still uncertain."),
        ("What is the most beautiful language in the world?", "Beauty in language is entirely in the ear of the beholder! Italian and French are praised for musical vowels, Arabic for rhythmic poetry, and Japanese for melodic cadence."),
        ("How will the stock market perform next quarter?", "Short-term market movements are inherently unpredictable due to macroeconomic shifts, investor sentiment, and unexpected events. Most financial advisors recommend long-term diversification over trying to time quarterly swings."),
        ("Which smartphone brand takes the best photos?", "It shifts with every flagship release! iPhones generally lead in natural color science and video stabilization, Google Pixels excel at low-light computational HDR, and Samsung offers unmatched telephoto zoom."),
        ("What is the best age to learn a musical instrument?", "Any age! Children have neuroplasticity on their side, but adults learn faster conceptually and have better discipline and patience."),
        ("What's the best way to travel around Europe?", "It depends on your travel style and itinerary! High-speed trains (like Eurail) are scenic, comfortable, and connect city centers directly, while budget flights are faster for cross-continent hops, and rental cars offer maximum rural flexibility."),
        ("What is the hardest programming language to master?", "Low-level languages like C++ and Rust are notorious for complex memory management and borrow checkers, while esoteric languages like Malbolge are deliberately obtuse. Among mainstream languages, Haskell and C++ frequently top developer difficulty surveys."),
        ("Will electric cars completely replace gas cars by 2035?", "Adoption is accelerating dramatically with policy mandates and cheaper battery tech, but infrastructure disparities, charging accessibility in rental housing, and commercial transport needs make a total global phase-out by 2035 ambitious and geographically uneven."),
        ("What is the best film ever made?", "Film critics and cinephiles often divide between masterpieces like Citizen Kane, The Godfather, Vertigo, and Tokyo Story. The answer largely hinges on whether you prize technical innovation, emotional storytelling, or thematic depth.")
    ]
    records = []
    prefixes = ["", "Tell me honestly: ", "In your opinion, ", "Do you know ", "I've been wondering, ", "What do you think: ", "Be honest: ", "Just curious: ", "Realistically, "]
    seen = set()
    for u_base, a in uncertainty_pairs:
        for p in prefixes:
            u = f"{p}{u_base}".strip()
            h = f"{u}|{a}"
            if h not in seen and len(records) < 110:
                seen.add(h)
                records.append({
                    "source": "curated_greetings_casual",
                    "category": "general_conversation",
                    "subtype": "uncertainty",
                    "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                })
    return records[:110]

def build_category_7_clarifications() -> List[Dict[str, Any]]:
    """120 unique records asking clarifying questions."""
    clarify_pairs = [
        ("Can you help me fix this?", "I'd be happy to! What are you trying to fix—is it a snippet of code, a piece of writing, or something mechanical? Feel free to share the details."),
        ("I need a recommendation.", "Sure thing! What kind of recommendation are you looking for—a book, a software tool, a movie, or something to cook?"),
        ("How do I make it faster?", "To give you the best advice, what specific system or process are you trying to speed up? For instance, is it a Python function, a web page, database queries, or a computer startup?"),
        ("What should I cook tonight?", "Let's figure it out! What ingredients do you already have on hand, and are you in the mood for something light, hearty, or super quick?"),
        ("I'm getting an error.", "Errors can be frustrating! What does the error message say, and what were you running when it popped up?"),
        ("I want to pick up a new hobby.", "That's exciting! Are you looking for something creative (like sketching or writing), hands-on (like baking or gardening), or more physically active?"),
        ("Can you make this sound better?", "Definitely! Go ahead and paste the text, and let me know if you want it more casual, concise, persuasive, or professional."),
        ("Help me pick a vacation destination.", "Fun project! Are you craving relaxation on a beach, mountain hiking, cultural city exploration, or a foodie road trip? And roughly what time of year?"),
        ("Can you summarize this for me?", "I'd be glad to! Paste the text or article, and let me know if you want a one-sentence summary, bullet points, or a quick paragraph overview."),
        ("I need an idea for a birthday gift.", "I can help brainstorm! Who is the gift for, what are their interests, and what budget range are you aiming for?"),
        ("How do I organize my day better?", "There are a few great methods! Are you struggling more with prioritization, losing track of tasks, or getting distracted while trying to focus?"),
        ("What exercises should I do?", "That depends on your goals! Are you looking to build strength, improve cardiovascular fitness, boost flexibility, or just stay active at home with no equipment?"),
        ("What should I wear today?", "I can give you some suggestions! What's the weather like outside, and are you dressing for work, an event, or just relaxing?"),
        ("Can you critique my resume?", "I'd be happy to review it! You can paste the text or the specific section you want feedback on, and tell me what role you're targeting."),
        ("Can you suggest a playlist?", "Sure! Are you looking for focus/instrumental background music, upbeat workout tunes, or something chill for evening relaxing?"),
        ("How do I fix my sleep schedule?", "To give you practical advice, are you struggling more with falling asleep at night, waking up in the middle of the night, or shifting hours after jet lag or shift work?")
    ]
    records = []
    prefixes = ["", "Hey, ", "Hi there, ", "Could you help me out? ", "Quick question: ", "I need some help. ", "Hello! ", "Hey assistant, ", "Can you help? "]
    seen = set()
    for u_base, a in clarify_pairs:
        for p in prefixes:
            u = f"{p}{u_base}".strip()
            h = f"{u}|{a}"
            if h not in seen and len(records) < 120:
                seen.add(h)
                records.append({
                    "source": "curated_greetings_casual",
                    "category": "general_conversation",
                    "subtype": "clarification",
                    "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                })
    return records[:120]

def build_category_8_proportionate_short() -> List[Dict[str, Any]]:
    """120 unique proportionate short responses."""
    short_pairs = [
        ("Cool.", "Glad you think so!"),
        ("Nice.", "Right? Pretty neat."),
        ("Yep.", "Sounds like we're on the same page."),
        ("Wait, really?", "Really! Surprising, isn't it?"),
        ("Makes sense.", "Glad that clicks."),
        ("Hmm.", "Take your time—let me know if you want to mull it over or explore another angle."),
        ("Interesting.", "It really is! Let me know if you want to dig deeper into why that happens."),
        ("Oh wow.", "Quite something, right?"),
        ("Huh, I didn't know that.", "Always fun to learn something new!"),
        ("I see.", "Feel free to ask if anything about it needs clarification."),
        ("Fair enough.", "Glad we're aligned!"),
        ("True.", "Definitely."),
        ("Gotcha.", "Glad that makes sense!"),
        ("Makes sense to me.", "Awesome!"),
        ("Totally.", "Indeed!"),
        ("Good to know.", "Happy to share!"),
        ("Fair point.", "Glad we see eye to eye on that."),
        ("Interesting angle.", "It really sheds a different light on it!"),
        ("Sounds good.", "Great! Let me know if you need anything else."),
        ("Understood.", "Perfect!"),
        ("Got it.", "Glad to hear!"),
        ("Sounds great.", "Awesome! Have fun with it.")
    ]
    records = []
    prefixes = ["", "Oh, ", "Okay, ", "Ah, ", "Alright, ", "Well, "]
    seen = set()
    for u_base, a in short_pairs:
        for p in prefixes:
            u = f"{p}{u_base}".strip()
            h = f"{u}|{a}"
            if h not in seen and len(records) < 120:
                seen.add(h)
                records.append({
                    "source": "curated_greetings_casual",
                    "category": "general_conversation",
                    "subtype": "short_reply",
                    "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                })
    return records[:120]

def build_category_9_topic_transitions() -> List[Dict[str, Any]]:
    """120 unique multi-turn conversational topic transitions."""
    transitions = [
        ("What's a good way to clean a cast iron skillet?", "Rinse it with warm water and use a stiff brush or coarse salt for stubborn food. Dry it immediately on the stove over low heat, then rub a drop of oil across the surface.",
         "Anyway, switching gears—how do noise-canceling headphones work?", "They use tiny external microphones to capture ambient background noise, then play an inverted sound wave (180 degrees out of phase) through the speakers, canceling out the unwanted noise through destructive interference."),
        ("Why does ice float on water?", "Water expands as it freezes into a crystalline structure, making ice roughly nine percent less dense than liquid water, so it floats to the surface.",
         "On a completely different note, what's a good book to read on a flight?", "'Born a Crime' by Trevor Noah or 'Project Hail Mary' by Andy Weir are great choices—both are gripping, fast-paced, and keep you engaged from takeoff to landing."),
        ("Is white chocolate actually chocolate?", "Technically, it doesn't contain cocoa solids, which give dark and milk chocolate their brown color and characteristic flavor. It is made from cocoa butter, sugar, and milk solids.",
         "Changing the subject for a second, do you know what causes hiccups?", "Hiccups are involuntary contractions of your diaphragm muscle, followed by a quick closure of your vocal cords, which produces the 'hic' sound. They're often triggered by eating too quickly, carbonated drinks, or sudden temperature changes."),
        ("What makes sourdough bread tangy?", "Lactic acid bacteria (lactobacilli) living symbiotically with wild yeasts in the starter ferment sugars into lactic and acetic acids, giving sourdough its pleasant tart flavor.",
         "Switching topics: how do airplanes generate lift?", "Airfoil-shaped wings force oncoming air downward and create lower pressure over the curved top of the wing compared to the bottom, producing an upward net force called lift."),
        ("What is the difference between a fruit and a vegetable botanically?", "Botanically, fruits develop from the flower of a plant and contain seeds (like tomatoes, cucumbers, and apples), whereas vegetables are other edible plant parts like roots, stems, or leaves (like carrots, celery, and spinach).",
         "On another topic, why do onions make you cry when you chop them?", "Cutting an onion breaks cell walls, allowing enzymes to mix and produce a volatile sulfur compound (syn-propanethial-S-oxide). When it reaches your eyes, sensory nerves trigger reflex tears to flush it out."),
        ("Why is the sky blue?", "Molecules in the atmosphere scatter sunlight in all directions, and blue light scatters more than other colors because it travels in smaller, shorter waves (Rayleigh scattering).",
         "By the way, what's an easy way to brew good coffee at home?", "A French press or pour-over cone is simple and forgiving. Use freshly ground beans, water just off the boil (around 200°F/93°C), and steep for about four minutes."),
        ("How do plants survive winter?", "Perennial plants slow their metabolism, drop leaves to prevent moisture loss, and concentrate cellular sugars and proteins which act as natural antifreeze in their tissues.",
         "Completely random question: what's the difference between baking powder and baking soda?", "Baking soda is pure sodium bicarbonate which requires an acid to activate, while baking powder already contains an acid salt and only needs moisture and heat to produce carbon dioxide bubbles."),
        ("What causes thunder?", "A lightning bolt superheats the surrounding air to around 30,000°C in microseconds. This rapid expansion creates a shock wave that compresses the air into the loud acoustic boom we hear as thunder.",
         "Shifting topics: what's a great beginner recipe to practice knife skills?", "A classic French mirepoix (diced onions, carrots, and celery) for soup or a vibrant pico de gallo salsa will give you tons of repetitive, satisfying dicing practice.")
    ]
    records = []
    intros = ["", "Hey! ", "Quick question: ", "Hi there, "]
    transition_phrases = [
        "", " Thanks for that!", " That makes sense.", " Awesome."
    ]
    seen = set()
    for q1, a1, q2, a2 in transitions:
        for i in intros:
            for t in transition_phrases:
                msgs = [
                    {"role": "user", "content": f"{i}{q1}".strip()},
                    {"role": "assistant", "content": a1},
                    {"role": "user", "content": f"{q2}{t}".strip()},
                    {"role": "assistant", "content": a2}
                ]
                h = compute_hash(msgs)
                if h not in seen and len(records) < 120:
                    seen.add(h)
                    records.append({
                        "source": "curated_greetings_casual",
                        "category": "general_conversation",
                        "subtype": "topic_transition",
                        "messages": msgs
                    })
    return records[:120]

def build_category_10_signoffs() -> List[Dict[str, Any]]:
    """110 unique ending conversations naturally."""
    signoffs = [
        ("I've got to run now, thanks!", "Have a great day! Catch you later."),
        ("Heading to bed, goodnight!", "Goodnight! Sleep well and recharge."),
        ("Time for my meeting, talk later!", "Good luck with your meeting! Talk later."),
        ("Catch you later!", "Catch you later! Take care."),
        ("That's all for today, take care!", "You too! Have a wonderful rest of your day."),
        ("See you around!", "See you around! Feel free to stop by anytime."),
        ("Bye for now!", "Bye! Wishing you the best with your projects."),
        ("Have a great weekend!", "Thanks! I hope you have a relaxing and fun weekend as well."),
        ("Signing off, thanks for the help.", "Anytime! Have a restful evening."),
        ("Talk to you tomorrow!", "Sounds good! See you tomorrow."),
        ("Gotta run, talk soon!", "Take care! Have a great day."),
        ("Off to dinner now, bye!", "Enjoy your dinner! Catch you later."),
        ("Thanks for everything, signing out.", "Anytime! Have a relaxing evening."),
        ("That's it for me today, see ya!", "See ya! Best of luck with everything."),
        ("Have a wonderful rest of your day!", "Thank you, you too! Take care."),
        ("Closing my laptop now, goodnight!", "Goodnight! Sleep well."),
        ("Talk to you later, bye!", "Bye! Take care until next time."),
        ("Logging off for the night, cheers!", "Cheers! Rest well."),
        ("Time for lunch, talk to you later!", "Enjoy your lunch! Have a great afternoon."),
        ("I'm heading out now, thanks again!", "You're very welcome! Safe travels and take care.")
    ]
    records = []
    prefixes = ["", "Alright, ", "Okay, ", "Well, ", "Thanks, ", "Great, "]
    seen = set()
    for u_base, a in signoffs:
        for p in prefixes:
            u = f"{p}{u_base}".strip()
            h = f"{u}|{a}"
            if h not in seen and len(records) < 110:
                seen.add(h)
                records.append({
                    "source": "curated_greetings_casual",
                    "category": "general_conversation",
                    "subtype": "sign_off",
                    "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                })
    return records[:110]

def generate_all_greetings_casual() -> List[Dict[str, Any]]:
    all_records = []
    all_records.extend(build_category_1_greetings())
    all_records.extend(build_category_2_acknowledgements())
    all_records.extend(build_category_3_checkins())
    all_records.extend(build_category_4_casual_qa())
    all_records.extend(build_category_5_multiturn_social())
    all_records.extend(build_category_6_uncertainty())
    all_records.extend(build_category_7_clarifications())
    all_records.extend(build_category_8_proportionate_short())
    all_records.extend(build_category_9_topic_transitions())
    all_records.extend(build_category_10_signoffs())
    
    # Exact deduplication check
    seen = set()
    unique = []
    for r in all_records:
        h = compute_hash(r["messages"])
        if h not in seen:
            seen.add(h)
            unique.append(r)
            
    rng = random.Random(42)
    rng.shuffle(unique)
    logger.info(f"Generated total unique records: {len(unique):,}")
    return unique

def main():
    tok = HuggingFaceBPEWrapper.load(repo_root / "modern_gpt/artifacts/tokenizer_16k_v2")
    serializer = ConversationSerializer(tok, max_length=4096)
    
    records = generate_all_greetings_casual()
    
    out_file = OUT_RAW / "greetings_casual_v1.jsonl"
    logger.info(f"Serializing and verifying {len(records)} records to {out_file}...")
    
    total_sup_tokens = 0
    total_in_tokens = 0
    subtypes = Counter()
    lengths = []
    
    verified_records = []
    for r in records:
        msgs = r["messages"]
        enc = serializer.encode_conversation(msgs)
        in_toks = len(enc["input_ids"])
        labels = enc["labels"]
        sup_toks = sum(1 for l in labels if l != -100)
        
        assert in_toks <= 4096, "Record exceeds 4096 context length!"
        assert sup_toks > 0, "Record has zero supervised tokens!"
        
        r["total_tokens"] = in_toks
        r["supervised_tokens"] = sup_toks
        
        total_in_tokens += in_toks
        total_sup_tokens += sup_toks
        subtypes[r.get("subtype", "unknown")] += 1
        lengths.append(sup_toks)
        verified_records.append(r)
        
    with open(out_file, "w", encoding="utf-8") as f:
        for r in verified_records:
            f.write(json.dumps(r) + "\n")
            
    # Compute sha256
    sha256_hash = hashlib.sha256()
    with open(out_file, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    pool_sha256 = sha256_hash.hexdigest()
    
    mean_sup = round(float(np.mean(lengths)), 2)
    median_sup = float(np.median(lengths))
    std_sup = round(float(np.std(lengths)), 2)
    min_sup = int(np.min(lengths))
    max_sup = int(np.max(lengths))
    
    manifest = {
        "manifest_name": "greetings_casual_v1_manifest",
        "version": "1.0.0",
        "phase": "1B.3A_supplementation",
        "random_seed": 42,
        "tokenizer_fingerprint": tok.fingerprint,
        "dataset_file": str(out_file.relative_to(repo_root)),
        "file_sha256": pool_sha256,
        "total_records": len(verified_records),
        "total_input_tokens": total_in_tokens,
        "total_supervised_tokens": total_sup_tokens,
        "mean_supervised_tokens": mean_sup,
        "median_supervised_tokens": median_sup,
        "std_supervised_tokens": std_sup,
        "min_supervised_tokens": min_sup,
        "max_supervised_tokens": max_sup,
        "subtype_distribution": dict(subtypes.most_common()),
        "validation_status": "PASSED"
    }
    
    manifest_file = OUT_MANIFESTS / "greetings_casual_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest to {manifest_file}")
    
    print("\n=== GREETINGS & CASUAL SUPPLEMENTATION COMPLETED ===")
    print(f"Total Records: {len(verified_records):,}")
    print(f"Total Supervised Tokens: {total_sup_tokens:,}")
    print(f"Mean Supervised Tokens: {mean_sup}")
    print(f"Median Supervised Tokens: {median_sup}")
    print(f"Min / Max Supervised Tokens: {min_sup} / {max_sup}")
    print(f"Subtypes Breakdown:")
    for st, c in subtypes.most_common():
        print(f"  {st:25s}: {c:5,d} records")
    print(f"SHA-256: {pool_sha256}")

if __name__ == "__main__":
    main()
