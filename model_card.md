# Model Card: Music Recommender Simulation

## Model Name

VibeMatch 1.0

## Goal / Task

This recommender tries to suggest songs a user might like.
It predicts which songs are the best fit based on genre, mood, and energy.
Its goal is to rank the small song catalog from best match to weakest match.

## Data Used

The dataset has 10 songs.
Each song includes genre, mood, energy, tempo, valence, danceability, and acousticness.
The user profile uses favorite genre, favorite mood, target energy, and a likes_acoustic field.
The dataset is very small, so it does not represent many kinds of music or many kinds of listeners.
Some features are stored in the data but are not used in the current score.

## Algorithm Summary

The system looks at one song at a time.
It gives points if the song's genre matches the user's favorite genre.
It gives points if the song's mood matches the user's favorite mood.
It also gives more points when the song's energy is close to the user's target energy.
Right now, energy matters more than genre or mood, so it has the biggest effect on the final score.

## Observed Behavior / Biases

One clear pattern is that the system now favors energy very strongly.
Songs with similar energy can rank high even if their genre or mood does not really fit the user.
This can create a filter bubble where the user keeps seeing songs with similar intensity.
The system also struggles with mixed or conflicting preferences because it only uses one genre and one mood at a time.

## Evaluation Process

I tested the system with several user profiles in the terminal.
I used profiles like High-Energy Pop, Chill Lofi, and Deep Intense Rock.
I also tested edge cases, like conflicting preferences and extreme energy values.
Then I compared the top 5 results for each profile and checked whether the rankings felt reasonable.

## Intended Use and Non-Intended Use

This system is designed for classroom learning and simple experiments.
It is good for showing how recommendation rules turn user preferences into rankings.
It is not meant for real users, music apps, or high-stakes decisions.
It should not be used as a serious measure of taste, fairness, or user satisfaction.

## Ideas for Improvement

- Use more features like acousticness, valence, and danceability in the score.
- Support more complex users with multiple genres, moods, or weighted preferences.
- Add diversity rules so the top results are not all the same type of song.

## Personal Reflection

My biggest learning moment was seeing how much the recommendations changed when I changed just a few scoring weights.
That helped me understand how even a simple system can shape what users see most often.
AI tools helped me move faster when I was testing ideas, writing profile experiments, and checking how the code behaved.
I still had to double-check the results, because sometimes a suggestion sounded good in theory but did not really match the output in the terminal.
I was surprised that such a simple algorithm could still feel like a real recommender when the top songs matched the user's vibe.
If I extended this project, I would try adding more user preferences and making the recommendations more diverse.
