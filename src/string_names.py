fake_str = "fake"
genuine_str = "genuine"

original_radio_str = (
    "<p> Determine if the review below is predicted "
    + genuine_str
    + " or "
    + fake_str
    + ' by the AI system  (Can only select once) </p> \n<input type="radio" name="options" id="option1"\
 value="genuine" onchange="doneSelect();"> '
    + genuine_str
    + ' </input><br>\n<input type="radio"\
  name="options" id="option2" value="fake" onchange="doneSelect();"> '
    + fake_str
    + " </input><br><br>"
)
# radio_str=''
genuine_correct_str = (
    "<p>Your guess was correct! The AI system had initially predicted "
    + genuine_str
    + ". <br>\
 </p>"
)
genuine_incorrect_str = (
    "<p>Your guess was incorrect! The AI system had initially predicted "
    + fake_str
    + " but you guessed "
    + genuine_str
    + ". <br>  </p>"
)
fake_incorrect_str = (
    "<p>Your guess was incorrect! The AI system had initially predicted "
    + genuine_str
    + " but you guessed "
    + fake_str
    + ". <br>  </p>"
)
fake_correct_str = (
    "<p>Your guess was correct! The AI system had initially predicted "
    + fake_str
    + ". <br>  </p>"
)
# edit_review_genuine = (
#     "<p id='Editing'>Please try editing the review so that the AI system predicts "
#     + genuine_str
#     + ". Note that the AI system outputs, confidence and explanations update after 3 seconds of\
#  the last edit, or upon pressing Shift+Enter.</p>"
# )
# edit_review_fake = (
#     "<p id='Editing'>Please try editing the review so that the AI system predicts "
#     + fake_str
#     + ". Note that the AI system outputs, confidence and explanations update after 3 seconds of the last edit,\
#  or upon pressing Shift+Enter.</p>"
# )

edit_review_genuine = (
    "<p id='Editing'>Please try editing the review so that the AI system predicts "
    + genuine_str
    + ". Note that the AI system outputs and confidence update after 3 seconds of\
 the last edit, or upon pressing Shift+Enter.</p>"
)
edit_review_fake = (
    "<p id='Editing'>Please try editing the review so that the AI system predicts "
    + fake_str
    + ". Note that the AI system outputs and confidence update after 3 seconds of the last edit,\
 or upon pressing Shift+Enter.</p>"
)

start_page_error_message = "Please accept the Terms and Conditions before proceeding."
end_page_error_message = "Please answer all the survey questions."
div_editable_str = (
    '<div  addToForm="f" id="div_border" name="div_border" contenteditable="true">'
)
div_readonly_str = '<div  addToForm="f" id="div_border" name="div_border">'
successful_flip_attempt = '<p id="Editing"><span style="background-color: rgb(0, 255,0)">\
 Congrats, you have been able to flip AI system prediction. You can move to the next example by\
  clicking the Next button. </span></p>'
unsuccessful_flip_attempt = ""
already_successful_flip_attempt = '<span style="background-color: rgb(0, 255,0)">\
 You have already flipped AI system prediction</span>'
heading_guess_str = "Can you guess the AI system outcome?"
heading_edit_str = "Please edit the review"

progress_str = "Most confidence reduced so far: "
last_step_str = "Confidence reduced in last attempt: "
# last_step_neg_str="your last attempt increased model confidence by "
attention_explain_image = '<img src="static/color_gradient_image.png" width="80%">'
incorrect_edit_str = '<p><span style="background-color: rgb(255,0, 0)">\
 Your previous edit violated our instructions by adding or removing information relevant to hotel experience.</span></p>'
next_viewable_str = '<input name="submit_btn" type="submit" value="Next" id="Next"/>'
next_hidden_str = '<input name="submit_btn" type="submit" value="Next" id="Next" style="display:none;"/>'

global_logreg_explanation_str = '<div class="float-container">\
            \
              <div class="float-child">\
                <div class="green">\
                  <h2> Words associated with Genuine </h2>\
                  <span style="background-color: rgb(255, 255, 0)"> small</span><br>\
                  <span style="background-color: rgb(255, 255, 0)"> floor</span><br>\
                  <span style="background-color: rgb(255, 255, 0)"> location</span><br>\
                  <span style="background-color: rgb(255, 255, 0)"> on</span><br>\
                  <span style="background-color: rgb(255, 255, 0)"> times</span><br>\
                  <span style="background-color: rgb(255, 255, 0)"> priceline</span><br>\
                  <span style="background-color: rgb(255, 255, 0)"> window</span><br>\
                  <span style="background-color: rgb(255, 255, 0)"> conference</span><br>\
                  <span style="background-color: rgb(255, 255, 0)"> rate</span><br>\
                  <span style="background-color: rgb(255, 255, 0)"> breakfast</span><br>\
                </div>\
              </div>\
              \
              <div class="float-child">\
                <div class="blue">\
                  <h2> Words associated with Fake </h2>\
                  <span style="background-color: rgb(255, 0, 255)"> chicago</span><br>\
                  <span style="background-color: rgb(255, 0, 255)"> my</span><br>\
                  <span style="background-color: rgb(255, 0, 255)"> luxury</span><br>\
                  <span style="background-color: rgb(255, 0, 255)"> luxurious</span><br>\
                  <span style="background-color: rgb(255, 0, 255)"> definitely</span><br>\
                  <span style="background-color: rgb(255, 0, 255)"> hotel</span><br>\
                  <span style="background-color: rgb(255, 0, 255)"> staying</span><br>\
                  <span style="background-color: rgb(255, 0, 255)"> experience</span><br>\
                  <span style="background-color: rgb(255, 0, 255)"> vacation</span><br>\
                  <span style="background-color: rgb(255, 0, 255)"> ever</span><br>\
                </div>\
              </div>\
              \
            </div>\
            <br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>'
global_bert_explanation_str = '<h4> Words associated with Genuine - </h4> \
                  <span style="background-color: rgb(255, 255, 74)"> Location</span>, \
                  <span style="background-color: rgb(255, 255, 112)"> floor</span>, \
                  <span style="background-color: rgb(255, 255, 120)"> elevators</span>, \
                  <span style="background-color: rgb(255, 255, 127)"> location</span>, \
                  <span style="background-color: rgb(255, 255, 128)"> elevator</span>, \
                  <span style="background-color: rgb(255, 255, 129)"> bell</span>, \
                  <span style="background-color: rgb(255, 255, 130)"> large</span>, \
                  <span style="background-color: rgb(255, 255, 133)"> bar</span>, \
                  <span style="background-color: rgb(255, 255, 135)"> 2nd</span>, \
                  <span style="background-color: rgb(255, 255, 140)"> rate</span>, \
                  <span style="background-color: rgb(255, 255, 143)"> /</span>, \
                  <span style="background-color: rgb(255, 255, 144)"> .</span>, \
                  <span style="background-color: rgb(255, 255, 153)"> 2</span>, \
                  <span style="background-color: rgb(255, 255, 154)"> (</span>, \
                  <span style="background-color: rgb(255, 255, 156)"> River</span>, \
                  <span style="background-color: rgb(255, 255, 158)"> upgraded</span>, \
                  <span style="background-color: rgb(255, 255, 160)"> straight</span>, \
                  <span style="background-color: rgb(255, 255, 160)"> Book</span>, \
                  <span style="background-color: rgb(255, 255, 162)"> )</span>, \
                  <span style="background-color: rgb(255, 255, 164)"> upgrade</span>, \
                  <br>\
                <h4> Words associated with Fake - </h4> \
                  <span style="background-color: rgb(255, 0, 255)"> Regency</span>, \
                  <span style="background-color: rgb(255, 22, 255)"> luxury</span>, \
                  <span style="background-color: rgb(255, 27, 255)"> Chicago</span>, \
                  <span style="background-color: rgb(255, 114, 255)"> luxurious</span>, \
                  <span style="background-color: rgb(255, 153, 255)"> I</span>, \
                  <span style="background-color: rgb(255, 155, 255)"> uneven</span>, \
                  <span style="background-color: rgb(255, 163, 255)"> welcomed</span>, \
                  <span style="background-color: rgb(255, 166, 255)"> definitely</span>, \
                  <span style="background-color: rgb(255, 167, 255)"> People</span>, \
                  <span style="background-color: rgb(255, 170, 255)"> sleep</span>, \
                  <span style="background-color: rgb(255, 174, 255)"> Hotel</span>, \
                  <span style="background-color: rgb(255, 177, 255)"> heart</span>, \
                  <span style="background-color: rgb(255, 177, 255)"> And</span>, \
                  <span style="background-color: rgb(255, 179, 255)"> personally</span>, \
                  <span style="background-color: rgb(255, 180, 255)"> supposed</span>, \
                  <span style="background-color: rgb(255, 180, 255)"> seemed</span>, \
                  <span style="background-color: rgb(255, 181, 255)"> securing</span>, \
                  <span style="background-color: rgb(255, 183, 255)"> managed</span>, \
                  <span style="background-color: rgb(255, 184, 255)"> Hilton</span>, \
                  <span style="background-color: rgb(255, 185, 255)"> Help</span>, \
                  <br><br>\
                Note: this is not an exhaustive list and only contains top-20 words that are most associated with fake or genuine reviews.\
                <br><br>'
