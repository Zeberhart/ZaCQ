import React, { useState, useEffect, useRef} from 'react';
import Modal from 'react-bootstrap/Modal';
import Form from 'react-bootstrap/Form';
import Switch from '@material-ui/core/Switch';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Button from '@material-ui/core/Button';


function Settings(props) {
  return (
    <Modal
      {...props}
      size="lg"
      aria-labelledby="contained-modal-title-vcenter"
      centered
    >
      <Modal.Header closeButton>
        <Modal.Title id="contained-modal-title-vcenter">
          Settings
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
          <p>Your session ID is <span style={{fontWeight:"bold"}}>{props.user}</span></p>
          <p>Log search session:</p>
              <ToggleButtonGroup
              color="primary"
              value={props.logging}
              onChange={(e, l) => {if(l!==null){props.setLogging(l)}}}
              exclusive
            >
              <ToggleButton value={false}>Not Logging</ToggleButton>
              <ToggleButton value={true}>Logging</ToggleButton>
            </ToggleButtonGroup>
          <br/>
          <br/>
          <p>Search mode:</p>
            <ToggleButtonGroup
              color="primary"
              value={props.mode}
              onChange={(e, mode) => {if(mode!==null){props.setMode(mode)}}}
              exclusive
            >
              <ToggleButton value="cq">Mode 1</ToggleButton>
              <ToggleButton value="kw">Mode 2</ToggleButton>
            </ToggleButtonGroup>
      </Modal.Body>
      <Modal.Footer>
        <Button onClick={props.onHide}>Close</Button>
      </Modal.Footer>
    </Modal>
  );
}

export default Settings

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