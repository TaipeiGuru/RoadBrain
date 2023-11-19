import './Background.css'
import video1 from '../../Assets/video1.mp4'
import img1 from '../../Assets/img1.jpg'
import img2 from '../../Assets/img2.jpg'
import img3 from '../../Assets/img4.jpg'

const Background = ({playStatus, heroCount}) => {
    if(playStatus) {
        return (
            <video className="background fade-in" autoPlay loop muted>
                <source src={video1} type='video/mp4'/>
            </video>
        )
    }
    else if(heroCount==0){
        return <img src={img1} className='background' alt="" />
    }
    else if(heroCount==1){
        return <img src={img2} className='background' alt="" />
    }
    else if(heroCount==2){
        return <img src={img3} className='background' alt="" />
    }
}
  

export default Background
