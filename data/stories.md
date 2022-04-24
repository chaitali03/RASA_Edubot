## greet path
* greet
  - utter_greet

## summary path1
* summary{"topic":"science"}
  - action_tell_definition

## summary path2
* summary{"topic":"science"}
  - action_tell_definition
* positive
  - action_give_resource

## summary path3
* summary{"topic":"science"}
  - action_tell_definition
 * negative
  - utter_goodbye

## cummulativesummary path
* cumulativesummary
  - action_tell_cummulative_summary

## feedback path
* feedback
  - utter_feedback
* take_feedback
  - utter_take_feedback
  
## test path
* test
  - utter_test

## say goodbye
* goodbye
  - utter_goodbye
