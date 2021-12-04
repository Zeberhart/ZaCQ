import React, { useState, useEffect, useRef} from 'react';
// import ReactDOM from 'react-dom';

import PuffLoader from "react-spinners/PuffLoader";
import ReactPaginate from 'react-paginate';
import Button from 'react-bootstrap/Button';
import IconButton from "@material-ui/core/IconButton";
import SettingsIcon from '@material-ui/icons/Settings';
import Tooltip from '@material-ui/core/Tooltip';

import { generateId } from 'zoo-ids';

import './App.css';
import Searchbar from "./Searchbar"
import Result from "./Result"
import PrioritizeMessage from "./PrioritizeMessage"
import Settings from "./Settings"
import SubmitUI from "./SubmitUI"
import {CQ, KW} from "./Refinement"

function Search(props){
   
    const modes = ["cq", "kw"]
    const [user, setUser] = useState(generateId(Date.now()));
    const [mode, setMode] = useState(modes[0]);
    const [logging, setLogging] = useState(false);
    const [settingsShow, setSettingsShow] = React.useState(false);
    const [submitUIShow, setSubmitUIShow] = React.useState(false);

    const c_ws = useRef(null);
    const [c_connected, c_setConnected] = useState(false);
    const s_ws = useRef(null);
    const [s_connected, s_setConnected] = useState(false);

    const [ready, setReady] = useState(false);
    const [loading, setLoading] = useState(false);

    const perPage = 10
    const [page, setPage] = useState(0);
    
    const [input, setInput] = useState("");
    const [lastSearch, setLastSearch] = useState("");

    const [newData, setNewData] = useState(null);
    const [results, setResults] = useState([]);
    const [data, setData] = useState([]);
    const [CQData, setCQData] = useState(null);
    const [KWData, setKWData] = useState(null);

    const [history, setHistory] = useState([]);
    const [setValues, setSetValues] = useState({});
    const [intents, setIntents] = useState([]);
    const [candidates, setCandidates] = useState([]);
    const [notValues, setNotValues] = useState([]);
    const [notIntents, setNotIntents] = useState([]);
    const [rejects, setRejects] = useState([]);
   
    //Initiate connections to servers
    useEffect(() => {
      function processMessage(message){
          const data = JSON.parse(message.data)
          setNewData(data)
      }
        
      function connect(ws, path, callback){
          ws.current = new WebSocket(path);  

          ws.current.onopen = e => {
            console.log("ws open");
            callback(true)
          };

          ws.current.onclose = e => {
            callback(false)
            console.log('Socket is closed. Reconnect will be attempted in 1 second.', e.reason);
            setTimeout(function() {
              connect(ws, path, callback);
            }, 5000);
          };    

          ws.current.onmessage = message => {
              processMessage(message)  
          };

          ws.current.onerror = err => {
            console.error('Socket encountered error: ', err.message, 'Closing socket');
            ws.current.close();
          };
      }
        
      connect(s_ws, "ws://localhost:8002/feed", s_setConnected)
      connect(c_ws, "ws://localhost:8003/feed", c_setConnected)
      return () => {
          c_ws.current.close();
          s_ws.current.close();
      };
    }, []);

    //When connection statuses change, update ready state variable
    useEffect(() => {
        if(s_connected && c_connected){
            setReady(true)
            setLoading(false)
        }else{
            setReady(false)
        }
    }, [s_connected, c_connected]);

    //Process new data received from a server
    useEffect(() => {
        console.log(input)
        function noRefinements(){
            return !Object.keys(setValues).length && !notValues.length && !intents.length && !notIntents.length
        }
        if(!newData || !("type" in newData)){
          console.log("Received mystery message:")
          console.log(newData)
        }else{
          if(newData["type"] === "typing"){
              //Currently irrelevant, leftover from old project.
          }else if(newData["type"] === "results"){
            setResults(newData["indices"])
            //Not sure why it's *not* using stale state values
            sendMessage("search", "get-data", {"indices": newData["indices"]})
          }else if(newData["type"] === "data"){
            //Function definitions. Once retrieved, contact refinement server
            setData(newData["data"])
            if(mode==="cq"){
                //Not sure why it's *not* using stale state values
                const payload = {"indices": results, "docstrings":newData["data"].map(result=>result.docstring), "identifiers":newData["data"].map(result=>result.identifier), "set_values":setValues, "not_values":notValues, "query":lastSearch}
                sendMessage("clarify", "generate-cq", payload) 
            }else{
                //Not sure why it's *not* using stale state values
                const payload = {"indices": results,  "identifiers":newData["data"].map(result=>result.identifier), "docstrings":newData["data"].map(result=>result.docstring), "intents":intents, "not_intents":notIntents, "query":lastSearch} 
                sendMessage("clarify", "generate-kw", payload)
            }
          }else if(newData["type"] === "cq"){
            setCQData(newData)
            setKWData(null)
            setLoading(false)
          }else if(newData["type"] === "kw"){
            setKWData(newData)
            setCQData(null)
            setLoading(false)
          }else{
            console.log(newData)
          }
        }
    }, [newData]);
    
    //Scroll to top of the page on page change
    useEffect(() => {
      window.scrollTo({ top: 0, behavior: 'auto' });
    }, [page]);

    useEffect(() => {
      if(results.length){
          resetRefinement()
      }
    }, [mode]);
    
    
    /////////////////////////////////////
    /*------------- I/O -------------*/
    /////////////////////////////////////

    function sendMessage(server, type, content={}){
        if(ready){
            const message = JSON.stringify({type:type, mode:mode, ...content})
            if(server === "clarify"){
                c_ws.current.send(message)
            }else{
                s_ws.current.send(message)
            }
        }
    }

    /////////////////////////////////////
    /*------------- User Actions -------------*/
    /////////////////////////////////////

    function submitQuery(){
       sendMessage("search", "search", {query:input.trim(), text:input.trim()})
       setLastSearch(input.trim())
       resetCQState()
       resetKWState()
       setHistory([])
       wait()
    }

    
    function answerCQ(answer){
       updateHistory()
       if (answer!==null){
           acceptCQAnswer(answer)
       }else{
           rejectCQAnswer(answer)
       }
       setCQData(null)
       wait()
    }

    function answerKW(answer){
       console.log(answer)
       updateHistory()
       if (answer!==null){
           acceptKWAnswer(answer)
       }else{
           rejectKWAnswer(answer)
       }
       setKWData(null)
       wait()
    }

    function undoRefinement(){
       if(history.length){
           wait()
           const oldState = history[history.length-1]
           setRejects(oldState.rejects)
           setCandidates(oldState.candidates)
           setSetValues(oldState.setValues)
           setNotValues(oldState.notValues)
           setIntents(oldState.intents)
           setNotIntents(oldState.notIntents)
           setHistory(oldHistory=>oldHistory.filter((element, index) => index < oldHistory.length - 1));
           sendMessage("search", "search-rerank", {query:lastSearch, text:lastSearch, candidates:oldState.candidates, rejects:oldState.rejects})
        }
    }
    
    function resetRefinement(){
        wait()
        setHistory([])
        resetCQState()
        resetKWState()
        sendMessage("search", "search-rerank", {query:lastSearch, text:lastSearch, candidates:[], rejects:[]})
    }

    function changePage(newPage){
        setPage(newPage)
    }

    /////////////////////////////////////
    /*-------- Helper Functions -------*/
    /////////////////////////////////////

    function resetAll(){
       resetCQState()
       resetKWState()
       setHistory([])
       setData([])
       setInput("")
       setLastSearch("")
    }
    
    function resetCQState(){
       setCQData(null)
       setCandidates([])
       setSetValues({})
       setRejects([])
       setNotValues([])
    }
    
    function resetKWState(){
       setKWData(null)
       setCandidates([])
       setRejects([])
       setIntents([])
       setNotIntents([])
    }

    function changeMode(){
           if(mode == modes[0]){
                  setMode(modes[1])
           }else{
                  setMode(modes[0])
           }
    }
    
    function wait(){
       setPage(0)
       setLoading(true)
       window.scrollTo({ top: 0, behavior: 'auto' });
    }
    
    
    function updateHistory(){
       setHistory(oldHistory=>[...oldHistory, {candidates: candidates, rejects:rejects, setValues:setValues, notValues:notValues, intents:intents, notIntents:notIntents}])
    }
    
    function acceptCQAnswer(answer){
       const newCandidates = CQData.answers[answer]
       let update = {...CQData.inferred_values}
       setCandidates(newCandidates)
       if(CQData.target === "role"){
           const [verb, dobj, prep, pobj] = answer.split(",")
           if(verb){
               update["verb"]=verb
           }
           if(dobj){
               update["direct_object"]=dobj
           }
           if(prep){
               update["preposition"]=prep
           }
           if(pobj){
               update["preposition_object"]=pobj
           }
       }else if(CQData.target!==null){
           update[CQData.target]=answer
       }
       setSetValues({...setValues, ...update})
       sendMessage("search", "search-rerank", {query:lastSearch, text:lastSearch, candidates:newCandidates, rejects:rejects})
    }
    
    function rejectCQAnswer(answer){
       const newRejects = CQData.rejectables
       setRejects(newRejects)
       const newCandidates = CQData.reject_candidates
       setCandidates(newCandidates)
       let newNotValues = []
       if(CQData.target === "role"){
           for (const key of Object.keys(CQData.answers)) {
               const [verb, dobj, prep, pobj] = key.split(",")
               let currentNotValues = {"verb":verb}
               if(dobj){
                   currentNotValues["direct_object"] = dobj
               }else{
                   currentNotValues["preposition"] = prep
                   currentNotValues["preposition_object"] = pobj
               }
              newNotValues.push({...CQData.inferred_values, ...currentNotValues})     
           }
       }else if(CQData.target!==null){
           for (const key of Object.keys(CQData.answers)) {
               if(key && key!=="null"){
                  newNotValues.push({...CQData.inferred_values, [CQData.target]: key})
               }
           }
       }else{
          newNotValues.push(CQData.inferred_values)     
       }
       setNotValues(notValues.concat(newNotValues))
       sendMessage("search", "search-rerank", {query:lastSearch, text:lastSearch, candidates:newCandidates, rejects:newRejects})
    }
    
    function acceptKWAnswer(answer){
       const newCandidates = KWData.answers[answer]
       setCandidates(newCandidates)
       setIntents(intents.concat(answer))
       sendMessage("search", "search-rerank", {query:lastSearch, text:lastSearch, candidates:newCandidates, rejects:rejects})
    }
    
    function rejectKWAnswer(answer){
       const newRejects = KWData.rejectables
       setRejects(newRejects)
       const newCandidates = KWData.reject_candidates
       setCandidates(newCandidates)
       setNotIntents(notIntents.concat(Object.keys(KWData.answers)))
       sendMessage("search", "search-rerank", {query:lastSearch, text:lastSearch, candidates:newCandidates, rejects:newRejects})
    }

    function chooseResult(result){
        setSubmitUIShow(true)
//         resetAll()
    }
    
    return (
      <div className="px-5 pb-5 position-relative">
        <Settings
            show={settingsShow}
            onHide={() => setSettingsShow(false)}
            user={user}
            mode={mode}
            setMode={changeMode}
            logging={logging}
            setLogging={setLogging}
        />
                    
        <div className="w-100 px-3 sticky-top d-flex flex-column align-items-center" style={{top: -25, scrollBehavior:'auto'}}>
            <br/>
            <div className="position-absolute" style={{right:-50, top:27}}>
                <SettingsButton setSettingsShow={setSettingsShow}/>
            </div>
            <div 
                className="sticky-top  w-100" 
                style={{opacity:((!loading && ready)?1:.5), pointerEvents:((!loading && ready)?"auto":"none")}}
            >
                <Searchbar 
                    ready={(!loading && ready)} 
                    input={input} 
                    setInput={setInput} 
                    submitQuery={submitQuery} 
                />
            </div>
        </div>

        <div className="w-100 pb-2 px-3">
            <PrioritizeMessage 
                active={history.length>0} 
                numCandidates={candidates.length} 
                setValues={setValues} 
                intents={intents} 
                undoRefinement={undoRefinement} 
                resetRefinement={resetRefinement}
            />
        </div>
             
        {(!loading &&  ready) ?
            <div className="px-5">
                {(CQData && CQData.question) &&
                    <CQ {...CQData} answerCQ={answerCQ} setValues={setValues}/>
                }
                {(KWData && KWData.question) &&
                    <KW {...KWData} answerCQ={answerKW} intents={intents}/>
                }
                <div className="flex-1  ">
                    {data.slice(perPage*page, perPage*page+perPage).map((result, i)=>
                          <div className="pb-2">
                              <Result {...result} 
                                 num={1+i+(perPage*page)} 
                                 key={i}
                              /> 
                              {logging &&
                                   <div>
                               <br/>
                                  <SubmitButton chooseResult={chooseResult} result={result}/>
                                      </div>
                              }
                          </div>
                    )}
                </div>
                {data.length>0 &&
                    <div className="d-flex justify-content-center"> 
                        <ReactPaginate
                          previousLabel={'previous'}
                          nextLabel={'next'}
                          breakLabel={'...'}
                          breakClassName={'break-me'}
                          pageCount={results.length/perPage}
                          marginPagesDisplayed={2}
                          pageRangeDisplayed={5}
                          onPageChange={(data)=>{changePage(data.selected)} }
                          containerClassName={'pagination'}
                          pageClassName={"page-item"}
                          previousClassName={"page-item"}
                          nextClassName={"page-item"}
                          pageLinkClassName={"page-link"}
                          previousLinkClassName={"page-link"}
                          nextLinkClassName={"page-link"}
                          activeClassName={'active'}
                        />
                     </div>
                 }
             </div>
        :
            <div className="w-100" style={{ marginTop:'150px'}}>
                <PuffLoader 
                    css={"display: block; margin: 0 auto;"} 
                    color={!ready?"#D73636":"#36D7B7"} 
                    loading={true} 
                    speedMultiplier={.6} size={150}
                />
            </div>
        }
        
      </div>
  );
}


function SettingsButton(props){
    return(
        <Tooltip title="Settings">
            <IconButton 
                className="nofocus" 
                onClick={()=>(props.setSettingsShow(true))}
            >
              <SettingsIcon style={{fill: "#a4a4a4", fontSize:"1.5em"}}/>
            </IconButton>
          </Tooltip>
    )
}

function SubmitButton(props){
    return(
        <Button 
            className="nofocus" 
            onClick={()=>(props.chooseResult(props.result))}
        >
          ✅Submit
        </Button>
    )
}




export default Search