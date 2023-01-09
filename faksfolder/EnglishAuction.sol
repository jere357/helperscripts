// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./Auction.sol";

contract EnglishAuction is Auction {

    uint internal highestBid;
    uint internal initialPrice;
    uint internal biddingPeriod;
    uint internal lastBidTimestamp;
    uint internal minimumPriceIncrement;

    address internal highestBidder;

    constructor(
        address _sellerAddress,
        address _judgeAddress,
        Timer _timer,
        uint _initialPrice,
        uint _biddingPeriod,
        uint _minimumPriceIncrement
    ) Auction(_sellerAddress, _judgeAddress, _timer) {
        initialPrice = _initialPrice;
        biddingPeriod = _biddingPeriod;
        minimumPriceIncrement = _minimumPriceIncrement;
        // Start the auction at contract creation.
        lastBidTimestamp = time();
    }

    function bid() public payable {
        require(msg.value - highestBid >= minimumPriceIncrement && msg.value >= initialPrice);
        //ne svida mi se bas test za ovo al okej
        require(msg.value >= highestBid);
        require(time() < biddingPeriod);
        //refundaj pare outbiddanoj adresi
        address payable prevHighestBidder = payable(highestBidder);
        prevHighestBidder.transfer(highestBid);
        highestBid = msg.value;
        highestBidder = msg.sender;
        lastBidTimestamp = time();
        // TODO Your code here
    }

    function getHighestBidder() override public returns (address) {
        // TODO Your code here
        if(time() < biddingPeriod){
            return address(0);
        }
        //require(getAuctionOutcome() == Auction.Outcome.SUCCESSFUL);
        return highestBidder;
    }

    function enableRefunds() public {
        outcome = Auction.Outcome.NOT_SUCCESSFUL;
    }

}
