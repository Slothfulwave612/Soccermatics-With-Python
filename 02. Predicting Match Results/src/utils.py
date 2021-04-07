"""
__author__: Anmol_Durgapal(slothfulwave612)

Python module containing utility functions.
"""

import numpy as np


def get_home_away_df(df):
    """
    Function to get home and away dataframe.

    Args:
        df (pandas.DataFrame): required dataframe.

    Returns:
        tuple: containing home and away dataframe.
    """
    # drop unnecessary column
    df.drop("Unnamed: 0", inplace=True, axis=1)

    # make home dataframe
    home_df = df[
        df.columns[0:14].values
    ].copy()

    # make away dataframe
    away_df = df[
        np.concatenate([df.columns[0:1].values, df.columns[14:].values])
    ].copy()

    # rename and convert dtypes
    home_df = final_changes(home_df)
    away_df = final_changes(away_df)

    return home_df, away_df


def final_changes(df):
    """
    Function to rename and convert data-type
    in a dataframe.

    Args:
        df (pandas.DataFrame): required dataframe

    Returns:
        pandas.DataFrame: with renamed columns and converted dtypes.
    """
    # rename columns
    df.rename(
        mapper = dict(zip(df.columns, df.loc[0].values)),
        inplace=True, axis=1
    )

    # drop first row
    df.drop(0, axis=0, inplace=True)
    df = df.reset_index(drop=True)

    # convert data-types
    df[df.columns[1:9]] = df[df.columns[1:9]].astype(int)
    df[df.columns[9:]] = df[df.columns[9:]].astype(float)

    return df


def get_goals_projection(
    home_df, away_df, home_team, away_team, display_stats=False
):
    """
    Function to get the expected number of goals scored by
    the home and away team when they face each other.

    Args:
        home_df (pandas.DataFrame): dataframe with home data.
        away_df (pandas.DataFrame): dataframe with away data.
        home_team (str): home-team name.
        away_team (str): away-team name.
        display_stat (bool, optional): to display calculated statistics.
    
    Returns:
        tuple: containing expected goals for home and away team.
    """
    # average number of goals scored by home team
    avg_home_scored = home_df["GF"].sum() / home_df["MP"].sum()

    # average number of goals scored by away team
    avg_away_scored = away_df["GF"].sum() / away_df["MP"].sum()

    # average number of goals conceded by home team
    avg_home_conceded = home_df["GA"].sum() / home_df["MP"].sum()

    # average number of goals conceded by away team
    avg_away_conceded = away_df["GA"].sum() / away_df["MP"].sum()

    # get attach strength and pct-rating --> home team
    home_attack_strength, pct_rating_home_score = get_home_team_attack_strength(
        home_df, home_team, avg_home_scored
    )

    # get defense strength and pct-rating --> away team
    away_defense_strength, pct_rating_away_concede = get_away_team_defense_strength(
        away_df, away_team, avg_away_conceded
    )

    # get attach strength and pct-rating --> away team
    away_attack_strength, pct_rating_away_score = get_away_team_attack_strength(
        away_df, away_team, avg_away_scored
    )

    # get defense strength and pct-rating --> home team
    home_defensive_strength, pct_rating_home_concede = get_home_team_defense_strength(
        home_df, home_team, avg_home_conceded
    )

    # projecting expected home team goals
    home_goals_projected = home_attack_strength * away_defense_strength * avg_home_scored

    # projecting expected away team goals
    away_goals_projected = away_attack_strength * home_defensive_strength * avg_away_scored

    if display_stats:
        print_display_pct(
            home_team, away_team,
            round(pct_rating_home_score, 2), round(pct_rating_away_concede, 2),
            round(pct_rating_away_score, 2), round(pct_rating_home_concede, 2)
        )
        print(f"{home_team}'s Projected Goals: {home_goals_projected} | {away_team}'s Projected Goals: {away_goals_projected}")

    return home_goals_projected, away_goals_projected


def print_display_pct(
    home_team, away_team,
    pct_rating_home_score, pct_rating_away_concede,
    pct_rating_away_score, pct_rating_home_concede
):
    pct_list = [
        pct_rating_home_score, pct_rating_away_score,
        pct_rating_home_concede, pct_rating_away_concede,
    ]

    for i, pct in enumerate(pct_list):
        if i in [0, 1]:
            pick = "scores"
        else:
            pick = "concedes"

        if i % 2 == 0:
            team = home_team
            side = "home"
        else:
            team = away_team
            side = "away"

        if pct > 0:
            print(f"{team} {pick} {pct}% more goals at {side} than theoretical average league team.")
        elif pct == 0:
            print(f"{team} {pick} same number of goals at {side} as theoretical average league team.")
        else:
            print(f"{team} {pick} {-pct}% fewer goals at {side} than theoretical average league team.")

        print()


def get_home_team_attack_strength(
    home_df, home_team, avg_home_scored
):
    """
    Function to get home team attack strength.

    Args:
        home_df (pandas.DataFrame): dataframe with home data.
        home_team (str): home-team name.
        avg_home_scored (float): avg goals scored by home team.

    Returns:
        tuple: containing attack strength, percentage rating
    """
    # calculate goals scored at home by home-team
    goals_scored_home = home_df.loc[
        home_df["Squad"] == home_team, "GF"
    ].values[0]

    # calculate total number of matches played at home by home-team
    match_played_home = home_df.loc[
        home_df["Squad"] == home_team, "MP"
    ].values[0]

    # home team's average goals per home game
    home_team_per_home = goals_scored_home / match_played_home

    # home team's attack strength
    home_attack_strength = home_team_per_home / avg_home_scored

    # calculate percentage rating
    pct_rating_home_score = (home_team_per_home - avg_home_scored) / avg_home_scored * 100

    return home_attack_strength, pct_rating_home_score


def get_away_team_defense_strength(
    away_df, away_team, avg_away_conceded
):
    """
    Function to get away team defense strength.

    Args:
        away_df (pandas.DataFrame): dataframe with away data.
        away_team (str): away-team name.
        avg_away_conceded (float): avg goals conceded by away team.

    Returns:
        tuple: containing defense strength, percentage rating
    """
    # goals conceded away by away-team
    goals_conceded_away = away_df.loc[
        away_df["Squad"] == away_team, "GA"
    ].values[0]

    # total number of matches played away by away-team
    match_played_away = away_df.loc[
        away_df["Squad"] == away_team, "MP"
    ].values[0]

    # away team's average goals conceded per away game
    away_team_per_away = goals_conceded_away / match_played_away

    # away team's defensive strength
    away_defense_strength = away_team_per_away / avg_away_conceded

    # calculate percentage rating
    pct_rating_away_concede = (away_team_per_away - avg_away_conceded) / avg_away_conceded * 100

    return away_defense_strength, pct_rating_away_concede


def get_away_team_attack_strength(
    away_df, away_team, avg_away_scored
):
    """
    Function to get away team attack strength.

    Args:
        away_df (pandas.DataFrame): dataframe with away data.
        away_team (str): away-team name.
        avg_away_scored (float): avg goals scored by away team.

    Returns:
        tuple: containing attack strength, percentage rating
    """
    # goals scored by away-team
    goals_scored_away = away_df.loc[
        away_df["Squad"] == away_team, "GF"
    ].values[0]

    # total number of matches played away by away-team
    match_played_away = away_df.loc[
        away_df["Squad"] == away_team, "MP"
    ].values[0]

    # away team's average goals scored per away game
    away_team_per_away_scored = goals_scored_away / match_played_away

    # away team's attack strength
    away_attack_strength = away_team_per_away_scored / avg_away_scored

    # calculate percentage rating
    pct_rating_away_score = (away_team_per_away_scored - avg_away_scored) / avg_away_scored * 100

    return away_attack_strength, pct_rating_away_score


def get_home_team_defense_strength(
    home_df, home_team, avg_home_conceded
):
    """
    Function to get home team defense strength.

    Args:
        home_df (pandas.DataFrame): dataframe with home data.
        home_team (str): home-team name.
        avg_home_conceded (float): avg goals conceded by home team at home

    Returns:
        tuple: containing defense strength, percentage rating
    """
    # goals conceded by home-team
    goals_conceded_home = home_df.loc[
        home_df["Squad"] == home_team, "GA"
    ].values[0]

    # calculate total number of matches played at home by home-team
    match_played_home = home_df.loc[
        home_df["Squad"] == home_team, "MP"
    ].values[0]

    # home team's average goals conceded per home game
    home_team_per_home_conceded = goals_conceded_home / match_played_home

    # home team's defensive strength
    home_defensive_strength = home_team_per_home_conceded / avg_home_conceded

    # calculate percentage rating
    pct_rating_home_concede = (home_team_per_home_conceded - avg_home_conceded) / avg_home_conceded * 100

    return home_defensive_strength, pct_rating_home_concede
