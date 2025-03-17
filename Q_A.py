import json

# qa dataset
qa_data = [
    {"question": "What is the largest planet?", "answer": "Jupiter"},
    {"question": "What is the currency of Japan?", "answer": "Yen"},
    {"question": "What is the main language spoken in Spain?", "answer": "Spanish"},
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the largest mammal?", "answer": "Whale"},
    {"question": "What is the capital of Italy?", "answer": "Rome"},
    {"question": "What is the largest ocean?", "answer": "Pacific"},
    {"question": "What is the chemical symbol for oxygen?", "answer": "O2"},
    {"question": "What is the capital of the USA?", "answer": "Washington"},
    {"question": "What is the main language spoken in Canada?", "answer": "English"},
    {"question": "What is the smallest continent?", "answer": "Australia"},
    {"question": "What is the capital of Australia?", "answer": "Canberra"},
    {"question": "What is the largest bird?", "answer": "Ostrich"},
    {"question": "What is the chemical symbol for carbon?", "answer": "C"},
    {"question": "What is the capital of China?", "answer": "Beijing"},
    {"question": "What is the main language spoken in Japan?", "answer": "Japanese"},
    {"question": "What is the largest continent?", "answer": "Asia"},
    {"question": "What is the currency of India?", "answer": "Rupee"},
    {"question": "What is the currency of the UK?", "answer": "Pound"},
    {"question": "What is the main ingredient in pizza dough?", "answer": "Flour"},
    {"question": "What is the chemical symbol for hydrogen?", "answer": "H2"},
    {"question": "What is the smallest ocean?", "answer": "Arctic"},
    {"question": "What is the main language spoken in Italy?", "answer": "Italian"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "What is the capital of Canada?", "answer": "Ottawa"},
    {"question": "What is the largest desert?", "answer": "Sahara"},
    {"question": "What is the main language spoken in Germany?", "answer": "German"},
    {"question": "What is the currency of France?", "answer": "Euro"},
    {"question": "What is the largest land animal?", "answer": "Elephant"},
    {"question": "What is the smallest bird?", "answer": "Hummingbird"},
    {"question": "What is the currency of Russia?", "answer": "Ruble"},
    {"question": "What is the largest reptile?", "answer": "Saltwater crocodile"},
    {"question": "What is the main language spoken in Brazil?", "answer": "Portuguese"},
    {"question": "What is the capital of Argentina?", "answer": "Buenos Aires"},
    {"question": "What is the smallest country?", "answer": "Vatican"},
    {"question": "What is the chemical symbol for iron?", "answer": "Fe"},
    {"question": "What is the capital of South Africa?", "answer": "Pretoria"},
    {"question": "What is the largest lake?", "answer": "Caspian"},
    {"question": "What is the hottest desert?", "answer": "Sahara"},
    {"question": "What is the main language spoken in Russia?", "answer": "Russian"},
    {"question": "What is the largest canyon?", "answer": "Grand Canyon"},
    {"question": "What is the smallest planet?", "answer": "Mercury"},
    {"question": "What is the largest sea?", "answer": "Philippine"},
    {"question": "What is the capital of Egypt?", "answer": "Cairo"},
    {"question": "What is the largest mountain?", "answer": "Everest"},
    {"question": "What is the main ingredient in sushi?", "answer": "Rice"},
    {"question": "What is the capital of Turkey?", "answer": "Ankara"},
    {"question": "What is the smallest fish?", "answer": "Paedocypris"},
    {"question": "What is the currency of Australia?", "answer": "Dollar"},
    {"question": "What is the largest bay?", "answer": "Bengal"},
    {"question": "What is the main ingredient in tea?", "answer": "Tea leaves"},
    {"question": "What is the largest river?", "answer": "Amazon"},
    {"question": "What is the capital of Thailand?", "answer": "Bangkok"},
    {"question": "What is the largest swamp?", "answer": "Pantanal"},
    {"question": "What is the smallest country in Africa?", "answer": "Seychelles"},
    {"question": "What is the largest glacier?", "answer": "Lambert"},
    {"question": "What is the main language spoken in Thailand?", "answer": "Thai"},
    {"question": "What is the largest peninsula?", "answer": "Arabian"},
    {"question": "What is the smallest star?", "answer": "Proxima"},
    {"question": "What is the largest atoll?", "answer": "Kiritimati"},
    {"question": "What is the largest rainforest?", "answer": "Amazon"},
    {"question": "What is the smallest tree?", "answer": "Dwarf Willow"},
    {"question": "What is the main ingredient in coffee?", "answer": "Coffee beans"},
    {"question": "What is the largest waterfall?", "answer": "Victoria"},
    {"question": "What is the largest island?", "answer": "Greenland"},
    {"question": "What is the hottest place on Earth?", "answer": "Death Valley"},
    {"question": "What is the largest mountain range?", "answer": "Himalayas"},
    {"question": "What is the largest plateau?", "answer": "Tibetan"},
    {"question": "What is the main language spoken in Kenya?", "answer": "Swahili"},
    {"question": "What is the largest gulf?", "answer": "Mexico"},
    {"question": "What is the capital of Peru?", "answer": "Lima"},
    {"question": "What is the largest waterfall in Africa?", "answer": "Victoria"},
    {"question": "What is the smallest planet in the Solar System?", "answer": "Mercury"},
    {"question": "Which mammal lays eggs?", "answer": "Platypus"},
    {"question": "Which metal is the best conductor of electricity?", "answer": "Silver"},
    {"question": "What is the longest-running Broadway show?", "answer": "The Phantom of the Opera"},
    {"question": "Which is the tallest building in the world?", "answer": "Burj Khalifa"},
    {"question": "What is the currency of South Korea?", "answer": "Won"},
    {"question": "Who was the first female astronaut in space?", "answer": "Valentina Tereshkova"},
    {"question": "Which is the only continent without an active volcano?", "answer": "Australia"},
    {"question": "What is the largest reptile in the world?", "answer": "Saltwater crocodile"},
    {"question": "Which artist painted 'The Persistence of Memory'?", "answer": "Salvador Dalí"},
    {"question": "Which chess piece can only move diagonally?", "answer": "Bishop"},
    {"question": "What is the name of the world’s largest cave?", "answer": "Son Doong Cave"},
    {"question": "What is the most abundant gas in the Earth's atmosphere?", "answer": "Nitrogen"},
    {"question": "Which planet has the shortest day?", "answer": "Jupiter"},
    {"question": "What is the process of plants making their food called?", "answer": "Photosynthesis"},
    {"question": "Who was the first person to reach the South Pole?", "answer": "Roald Amundsen"},
    {"question": "Which is the world's largest freshwater lake?", "answer": "Lake Superior"},
    {"question": "Which country has the most islands?", "answer": "Sweden"},
    {"question": "What is the national animal of Scotland?", "answer": "Unicorn"},
    {"question": "Which river is the longest in Europe?", "answer": "Volga River"},
    {"question": "Which planet is known for having a hexagonal storm?", "answer": "Saturn"},
    {"question": "What is the largest moon in the Solar System?", "answer": "Ganymede"},
    {"question": "Who invented the World Wide Web?", "answer": "Tim Berners-Lee"},
    {"question": "What is the name of the fastest land snake?", "answer": "Black mamba"},
    {"question": "What is the most common element in the universe?", "answer": "Hydrogen"},
    {"question": "Which sea separates Europe and Africa?", "answer": "Mediterranean Sea"},
    {"question": "What is the world's largest amphibian?", "answer": "Chinese giant salamander"},
    {"question": "Which organ in the human body regenerates itself?", "answer": "Liver"},
    {"question": "Which US state has the longest coastline?", "answer": "Alaska"},
    {"question": "Which company created the first smartphone?", "answer": "IBM"},
    {"question": "Which bird has the longest wingspan?", "answer": "Wandering albatross"},
    {"question": "Who was the first emperor of China?", "answer": "Qin Shi Huang"},
    {"question": "What is the longest-running TV series?", "answer": "The Simpsons"},
    {"question": "Which sport uses a shuttlecock?", "answer": "Badminton"},
    {"question": "Which is the largest hot desert?", "answer": "Sahara Desert"},
    {"question": "Which mountain range separates Europe and Asia?", "answer": "Ural Mountains"},
    {"question": "What is the most spoken language in Africa?", "answer": "Swahili"},
    {"question": "Which is the smallest continent?", "answer": "Australia"},
    {"question": "What is the scientific name for humans?", "answer": "Homo sapiens"},
    {"question": "What is the chemical symbol for helium?", "answer": "He"},
    {"question": "Who wrote the Harry Potter series?", "answer": "J.K. Rowling"},
    {"question": "Which country produces the most coffee?", "answer": "Brazil"},
    {"question": "What is the only metal that is liquid at room temperature?", "answer": "Mercury"},
    {"question": "Who is known as the father of modern computers?", "answer": "Alan Turing"},
    {"question": "Which land animal can open its mouth the widest?", "answer": "Hippo"},
    {"question": "Which was the first country to reach space?", "answer": "Soviet Union"},
    {"question": "What is the study of fungi called?", "answer": "Mycology"},
    {"question": "Which is the only mammal that can fly?", "answer": "Bat"},
    {"question": "What is the main ingredient in guacamole?", "answer": "Avocado"},
    {"question": "What is the rarest blood type?", "answer": "AB negative"},
    {"question": "Which is the longest river in North America?", "answer": "Missouri River"},
    {"question": "What is the national sport of India?", "answer": "Field hockey"},
    {"question": "Who painted 'The Last Supper'?", "answer": "Leonardo da Vinci"},
    {"question": "What is the smallest dog breed?", "answer": "Chihuahua"},
    {"question": "What is the national flower of China?", "answer": "Peony"},
    {"question": "Which fish is known for its ability to generate electricity?", "answer": "Electric eel"},
    {"question": "What is the hardest known natural material?", "answer": "Diamond"},
    {"question": "Which continent has the most volcanoes?", "answer": "Asia"},
    {"question": "Which ancient wonder was located in Egypt?", "answer": "Great Pyramid of Giza"},
    {"question": "Which planet is known as the Ice Giant?", "answer": "Neptune"},
    {
        "question": "What is the capital of the United States?",
        "answer": "Washington, D.C."
    },
    {
        "question": "Which organ in the human body produces insulin?",
        "answer": "Pancreas"
    },
    {
        "question": "Who is known as the father of modern physics?",
        "answer": "Albert Einstein"
    },
    {
        "question": "What is the name of the fairy in Peter Pan?",
        "answer": "Tinker Bell"
    },
    {
        "question": "Which gas is essential for human respiration?",
        "answer": "Oxygen"
    },
    {
        "question": "What is the national sport of Canada?",
        "answer": "Ice hockey"
    },
    {
        "question": "Which animal is known to have the longest lifespan?",
        "answer": "Bowhead whale"
    },
    {
        "question": "Which country is famous for the Great Wall?",
        "answer": "China"
    },
    {
        "question": "How many players are there in a basketball team on the court?",
        "answer": "Five"
    },
    {
        "question": "What is the square root of 169?",
        "answer": "13"
    },
    {
        "question": "Who discovered America in 1492?",
        "answer": "Christopher Columbus"
    },
    {
        "question": "Which element is necessary for bones and teeth?",
        "answer": "Calcium"
    },
    {
        "question": "Which animal is known to have the strongest bite force?",
        "answer": "Saltwater crocodile"
    },
    {
        "question": "What is the capital of Germany?",
        "answer": "Berlin"
    },
    {
        "question": "How many sides does a dodecagon have?",
        "answer": "12"
    },
    {
        "question": "Which ocean is the deepest?",
        "answer": "Pacific Ocean"
    },
    {
        "question": "What is the capital of Russia?",
        "answer": "Moscow"
    },
    {
        "question": "What is the freezing point of water in Fahrenheit?",
        "answer": "32°F"
    },
    {
        "question": "Which scientist developed the polio vaccine?",
        "answer": "Jonas Salk"
    },
    {
        "question": "Which is the second most spoken language in the world?",
        "answer": "Mandarin Chinese"
    },
    {
        "question": "Which fruit has the most vitamin C?",
        "answer": "Guava"
    },
    {
        "question": "What is the most common blood type?",
        "answer": "O positive"
    },
    {
        "question": "Who was the first President of the United States?",
        "answer": "George Washington"
    },
    {
        "question": "Which planet has the most moons?",
        "answer": "Saturn"
    },
    {
        "question": "How many stripes are on the American flag?",
        "answer": "13"
    },
    {
        "question": "Which is the longest-running animated TV show?",
        "answer": "The Simpsons"
    },
    {
        "question": "What is the national animal of Australia?",
        "answer": "Kangaroo"
    },
    {
        "question": "Which vitamin is produced when the body is exposed to sunlight?",
        "answer": "Vitamin D"
    },
    {
        "question": "Who wrote 'The Odyssey'?",
        "answer": "Homer"
    },
    {
        "question": "Which musical instrument has 88 keys?",
        "answer": "Piano"
    },
    {
        "question": "Which is the largest cat species in the world?",
        "answer": "Siberian tiger"
    },
    {
        "question": "Which is the fastest bird?",
        "answer": "Peregrine falcon"
    },
    {
        "question": "Who discovered radioactivity?",
        "answer": "Henri Becquerel"
    },
    {
        "question": "Which city is known as the 'Big Apple'?",
        "answer": "New York City"
    },
    {
        "question": "What is the primary ingredient in chocolate?",
        "answer": "Cocoa beans"
    },
    {
        "question": "Which month has 28 or 29 days?",
        "answer": "February"
    },
    {
        "question": "What is the capital of South Korea?",
        "answer": "Seoul"
    },
    {
        "question": "What is the national currency of the UK?",
        "answer": "Pound sterling"
    },
    {
        "question": "Who painted 'Starry Night'?",
        "answer": "Vincent van Gogh"
    },
    {
        "question": "Which is the world's smallest bird?",
        "answer": "Bee hummingbird"
    },
    {
        "question": "What does the Richter scale measure?",
        "answer": "Earthquake magnitude"
    },
    {
        "question": "What is the name of Sherlock Holmes' assistant?",
        "answer": "Dr. John Watson"
    },
    {
        "question": "Which mammal can fly?",
        "answer": "Bat"
    },
    {
        "question": "Which Greek god was known as the god of war?",
        "answer": "Ares"
    },
    {
        "question": "What is the capital of Spain?",
        "answer": "Madrid"
    },
    {
        "question": "What is the human body's smallest bone?",
        "answer": "Stapes (in the ear)"
    },
    {
        "question": "Which country gifted the Statue of Liberty to the USA?",
        "answer": "France"
    },
    {
        "question": "How many hearts does an octopus have?",
        "answer": "Three"
    },
    {
        "question": "What is the national flower of Japan?",
        "answer": "Cherry blossom"
    },
    {
        "question": "Which sea creature has three hearts?",
        "answer": "Octopus"
    },
    {
        "question": "Which country is known for the Amazon Rainforest?",
        "answer": "Brazil"
    },
    {
        "question": "What is the world's highest waterfall?",
        "answer": "Angel Falls"
    },
    {
        "question": "Who was the first female Prime Minister of the UK?",
        "answer": "Margaret Thatcher"
    },
    {
        "question": "What is the process of converting water into vapor called?",
        "answer": "Evaporation"
    },
    {
        "question": "Which planet is known as the Morning Star?",
        "answer": "Venus"
    },
    {
        "question": "What is the capital of Mexico?",
        "answer": "Mexico City"
    },
    {
        "question": "What is the main gas found in Earth's atmosphere?",
        "answer": "Nitrogen"
    },
    {
        "question": "Who invented the light bulb?",
        "answer": "Thomas Edison"
    },
    {
        "question": "Which is the coldest place on Earth?",
        "answer": "Antarctica"
    },
    {
        "question": "What is the national dish of Italy?",
        "answer": "Pasta"
    },
    {
        "question": "What is the speed limit of sound in air?",
        "answer": "Approximately 343 m/s"
    },
    {
        "question": "Which color is associated with royalty?",
        "answer": "Purple"
    },
    {
        "question": "Which is the largest land carnivore?",
        "answer": "Polar bear"
    },
    {
        "question": "What is the study of stars and planets called?",
        "answer": "Astronomy"
    },
    {
        "question": "Which river flows through Egypt?",
        "answer": "Nile River"
    },
    {
        "question": "What is the national sport of Japan?",
        "answer": "Sumo wrestling"
    },
    {
        "question": "Which is the longest bone in the human body?",
        "answer": "Femur"
    },
    {
        "question": "Which is the highest-grossing movie of all time?",
        "answer": "Avatar"
    },
    {
        "question": "What is the largest flower in the world?",
        "answer": "Rafflesia arnoldii"
    },
    {
        "question": "Which is the brightest planet in the night sky?",
        "answer": "Venus"
    },
    {
        "question": "What is the main component of Earth's core?",
        "answer": "Iron and nickel"
    },
    {"question": "What is the chemical symbol for water?", "answer": "H2O"},
    {"question": "What is the hardest natural material?", "answer": "Diamond"},
    {"question": "What is the most abundant gas in Earth's atmosphere?", "answer": "Nitrogen"},
    {"question": "What is the square root of 81?", "answer": "9"},
    {"question": "Which gas do humans exhale?", "answer": "Carbon dioxide"},
    {"question": "What is the main ingredient in bread?", "answer": "Flour"},
    {"question": "What is the strongest muscle in the human body?", "answer": "Jaw"},
    {"question": "Which sense is the most powerful in detecting danger?", "answer": "Hearing"},
    {"question": "Which color absorbs the most heat?", "answer": "Black"},
    {"question": "What is the main function of the heart?", "answer": "Pumping blood"},
    {"question": "Which metal is most commonly used in electrical wiring?", "answer": "Copper"},
    {"question": "What is the primary source of energy for the Earth?", "answer": "Sun"},
    {"question": "What state of matter is glass technically?", "answer": "Amorphous solid"},
    {"question": "Which part of the plant conducts photosynthesis?", "answer": "Leaf"},
    {"question": "Which organ is responsible for filtering blood?", "answer": "Kidney"},
    {"question": "What is the most common liquid on Earth?", "answer": "Water"},
    {"question": "Which element is needed for humans to breathe?", "answer": "Oxygen"},
    {"question": "What force pulls objects toward the center of the Earth?", "answer": "Gravity"},
    {"question": "Which part of the body contains the most bones?", "answer": "Hand"},
    {"question": "Which number is neither prime nor composite?", "answer": "1"},
    {"question": "Which type of energy is produced by a moving object?", "answer": "Kinetic energy"},
    {"question": "Which animal can regenerate lost limbs?", "answer": "Starfish"},
    {"question": "What is the main function of white blood cells?", "answer": "Fighting infections"},
    {"question": "Which type of wave carries sound?", "answer": "Longitudinal wave"},
    {"question": "Which part of the eye controls how much light enters?", "answer": "Pupil"},
    {"question": "Which simple machine is a ramp?", "answer": "Inclined plane"},
    {"question": "Which unit is used to measure electrical resistance?", "answer": "Ohm"},
    {"question": "Which planet has the strongest gravity?", "answer": "Jupiter"},
    {"question": "What is the main component of bones?", "answer": "Calcium"},
    {"question": "Which organ produces insulin?", "answer": "Pancreas"},
    {"question": "Which type of cloud is associated with thunderstorms?", "answer": "Cumulonimbus"},
    {"question": "Which form of energy is stored in food?", "answer": "Chemical energy"},
    {"question": "Which unit is used to measure force?", "answer": "Newton"},
    {"question": "Which part of the cell contains genetic material?", "answer": "Nucleus"},
    {"question": "Which liquid metal is used in thermometers?", "answer": "Mercury"},
    {"question": "What is the most common type of rock on Earth's surface?", "answer": "Igneous rock"},
    {"question": "Which part of the body produces red blood cells?", "answer": "Bone marrow"},
    {"question": "Which gas makes up most of the Sun?", "answer": "Hydrogen"},
    {"question": "Which unit is used to measure loudness?", "answer": "Decibel"},
    {"question": "What type of charge do electrons carry?", "answer": "Negative"},
    {"question": "Which color of light has the shortest wavelength?", "answer": "Violet"},
    {"question": "What happens to water when it freezes?", "answer": "It expands"},
    {"question": "Which simple machine is found in scissors?", "answer": "Lever"},
    {"question": "Which state of matter has a definite shape and volume?", "answer": "Solid"},
    {"question": "Which unit is used to measure frequency?", "answer": "Hertz"},
    {"question": "What is the main function of the lungs?", "answer": "Breathing"},
    {"question": "Which type of animal lays eggs but produces milk?", "answer": "Platypus"},
    {"question": "Which type of energy does a battery store?", "answer": "Chemical energy"},
    {"question": "Which body organ controls balance?", "answer": "Inner ear"},
    {"question": "Which unit is used to measure temperature in science?", "answer": "Kelvin"},
    {"question": "Which animal can sleep while standing?", "answer": "Horse"},
    {"question": "Which type of tree does not lose its leaves in winter?", "answer": "Evergreen"},
    {"question": "Which state of matter has no definite shape or volume?", "answer": "Gas"},
    {"question": "What is the most common natural fiber?", "answer": "Cotton"},
    {"question": "Which animal has the largest brain relative to its body size?", "answer": "Dolphin"},
    {"question": "Which part of the body is mainly responsible for digestion?", "answer": "Stomach"},
    {"question": "Which type of waves are used in radios?", "answer": "Radio waves"},
    {"question": "Which sense do humans rely on the most?", "answer": "Sight"},
    {"question": "What is the most common element in the human body?", "answer": "Oxygen"},
    {"question": "Which type of energy is associated with heat?", "answer": "Thermal energy"},
    {"question": "Which gas is responsible for the greenhouse effect?", "answer": "Carbon dioxide"},
    {"question": "Which layer of the Earth is liquid?", "answer": "Outer core"},
    {"question": "Which type of diet consists only of plant-based food?", "answer": "Vegan"},
    {"question": "What is the most common type of sugar?", "answer": "Glucose"},
    {"question": "Which organ in the body helps detoxify substances?", "answer": "Liver"},
    {"question": "Which type of bond holds water molecules together?", "answer": "Hydrogen bond"},
    {"question": "Which type of lens is used in magnifying glasses?", "answer": "Convex"},
    {"question": "Which part of a plant absorbs water?", "answer": "Roots"},
    {"question": "Which part of the ear helps detect sound?", "answer": "Cochlea"},
    {"question": "Which type of rock is formed from cooled lava?", "answer": "Igneous rock"},
    {"question": "Which type of teeth are used for cutting food?", "answer": "Incisors"},
    {"question": "Which type of circuit allows electricity to flow?", "answer": "Closed circuit"},
    {"question": "Which part of the plant makes seeds?", "answer": "Flower"},
    {"question": "Which type of animal has a backbone?", "answer": "Vertebrate"},
    {"question": "Which type of energy is produced by the sun?", "answer": "Solar energy"},
    {"question": "Which process allows plants to make food?", "answer": "Photosynthesis"},
    {"question": "Which type of rock is made from compressed layers?", "answer": "Sedimentary"},
    {"question": "Which type of wave travels faster in water than air?", "answer": "Sound wave"}

]

# save to Json file
with open("qa_dataset.json", "w", encoding="utf-8") as f:
    json.dump(qa_data, f, indent=4)

print("Dataset saved to qa_dataset.json")
