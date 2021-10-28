import React from 'react';


function CQ(props){    
 return(
    <div className="pb-2">
        <div>{props.question}</div>
        <div className="d-flex flex-row flex-wrap">
             {props.answers && Object.keys(props.answers).map((answer, i) =>
                  <div 
                      className="me-3"
                      style={{cursor:"pointer", color:"#0000ee", fontWeight:"bold"}}
                      onClick={()=>props.answerCQ(answer)}
                      key={i}
                  >
                     {props.target===null?"Yes":(props.target!=="role"?answer:answer.split(",").join(" "))}
                  </div>
             )}
             <div 
                className="me-3"
                style={{cursor:"pointer", color:"#0000ee"}}
                onClick={()=>props.answerCQ(null)}
             >
                {props.target===null?"No":"None of these"}
                
             </div>
        </div>
    </div>
 )
}

function KW(props){
 return(
    <div className="pb-2">
        <div>{props.question}</div>
        <div className="d-flex flex-row flex-wrap">
             {props.answers && Object.keys(props.answers).map((answer, i) =>
                  <div 
                      className="me-3"
                      style={{cursor:"pointer", color:"#0000ee", fontWeight:"bold"}}
                      onClick={()=>props.answerCQ(answer)}
                      key={i}
                  >
                     {answer}
                  </div>
             )}
             <div 
                className="me-3"
                style={{cursor:"pointer", color:"#0000ee"}}
                onClick={()=>props.answerCQ(null)}
             >
                None of these
             </div>
        </div>
    </div>
 )
}

export {KW, CQ}