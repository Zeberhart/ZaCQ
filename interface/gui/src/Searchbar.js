import React from 'react';
import InputBase from '@material-ui/core/InputBase';
import IconButton from "@material-ui/core/IconButton";
import Paper from '@material-ui/core/Paper';
import Tooltip from '@material-ui/core/Tooltip';
import SearchIcon from "@material-ui/icons/Search";
import CancelIcon from '@material-ui/icons/Cancel';
import { makeStyles } from '@material-ui/core/styles';

function Searchbar(props){

  function _submit(){
    if(props.input.trim().length > 0){
        props.submitQuery()
    }
  }


  function _handleInputChange(e){
    if(e){
      props.setInput(e.target.value);
    }
  }


  const classes = useStyles();

  return(
      <div className="py-2" style={{backgroundColor:"white"}}>
        <Paper elevation={props.ready?4:0} component="form" style={{width:"100%"}}>
          <div className={classes.root}>
            <InputBase
              className={classes.input}
              value = {props.input}
              autoComplete="off"
              inputProps={{ 'aria-label': 'search',}}
              placeholder="Search..."
              onChange={_handleInputChange}
              onKeyPress={(evt)=>{
                if(evt.key === 'Enter'){
                  evt.target.blur()
                  evt.preventDefault();
                  _submit()
                }
              }}
            />
                {props.input &&
                  <Tooltip title="Cancel">
                    <IconButton className="nofocus" size="medium" onClick={()=>(props.setInput(""))}>
                      <CancelIcon style={{fill: "#c8c8c8"}} fontSize="inherit"/>
                    </IconButton>
                  </Tooltip>
                }
            <div >
                <Tooltip  title="Submit">
                  <IconButton className="nofocus" size="medium" onClick={_submit}>
                    <SearchIcon fontSize="inherit"/>
                  </IconButton>
                </Tooltip>
            </div>
          </div>
        </Paper>
      </div>
  )
}

const useStyles = makeStyles((theme) => ({
  root: {
    padding: '2px 4px',
    display: 'flex',
    alignItems: 'center',
    width: "100%",
  },
  input: {
    marginLeft: theme.spacing(1),
    flex: 1,
  },
  iconButton: {
    padding: 10,
  },
  divider: {
    height: 28,
    margin: 4,
  },
  link:{
    color:"#001081",
    cursor:"pointer",
    "&:hover":{
      textDecoration:"underline"
    }
  }
}));

export default Searchbar;