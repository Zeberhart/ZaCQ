import React, { useState, useEffect, useRef} from 'react';
import Modal from 'react-bootstrap/Modal';
import Form from 'react-bootstrap/Form';
import Switch from '@material-ui/core/Switch';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';


function SubmitUI(props) {
    
  const [satisfies, setSatisfies] = useState(null)
  const [best, setBest] = useState(null)
  const [explanation, setExplanation] = useState("")
    
  function submitAnswer(){
      props.recordAction("submit-ratings", {satisfies: satisfies, best:best, explanation:explanation})
      props.onHide()
      setSatisfies(null)
      setBest(null)
      setExplanation("")
  }
    
  return (
    <Modal
      {...props}
      size="lg"
      aria-labelledby="contained-modal-title-vcenter"
    backdrop="static"
      centered
    >
      <Modal.Header>
        <Modal.Title id="contained-modal-title-vcenter">
          Rate Submission
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
          <p>How confident are you that this function satisfies the task criteria?</p>

            <ToggleButtonGroup
              color="primary"
              value={satisfies}
              onChange={(e, sat)=>{if(sat){setSatisfies(sat)}}}
              exclusive
            >
              <ToggleButton value={1}>1 (Not confident)</ToggleButton>
              <ToggleButton value={2}>2</ToggleButton>
              <ToggleButton value={3}>3</ToggleButton>
              <ToggleButton value={4}>4</ToggleButton>
              <ToggleButton value={5}>5 (Very confident)</ToggleButton>
            </ToggleButtonGroup>
          <br/>
          <br/>
          <p>How confident are you that this is the best function that the search engine has available to meet your criteria?</p>
            <ToggleButtonGroup
              color="primary"
              value={best}
              onChange={(e, b)=>{if(b){setBest(b)}}}
              exclusive
            >
              <ToggleButton value={1}>1 (Not confident)</ToggleButton>
              <ToggleButton value={2}>2</ToggleButton>
              <ToggleButton value={3}>3</ToggleButton>
              <ToggleButton value={4}>4</ToggleButton>
              <ToggleButton value={5}>5 (Very confident)</ToggleButton>
            </ToggleButtonGroup>
          <br/>
          <br/>
          <p>Please provide a brief (1-2 sentence) explanation of what this function does and why you chose it.</p>
          <TextField
              id="outlined-textarea"
              label="Explanation"
              placeholder="..."
              value={explanation}
              onChange={(event)=>setExplanation(event.target.value)}
              multiline
              maxRows={4}
              fullWidth
          />

      </Modal.Body>
      <Modal.Footer>
        <Button 
            onClick={submitAnswer} 
            disabled={!satisfies || !best}
        >
            Submit
        </Button>
      </Modal.Footer>
    </Modal>
  );
}

export default SubmitUI

//         <InputBase
//           value = {props.user}
//           autoComplete="off"
//           inputProps={{ 'aria-label': 'email',}}
//           placeholder="Email"
//           onChange={(e)=>props.setUser(e.target.value)}
//           onKeyPress={(evt)=>{
//             if(evt.key === 'Enter'){
//               evt.target.blur()
//               evt.preventDefault();
//             }
//           }}
//         />