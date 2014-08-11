#include <iostream>
#include <string>
#include <vector>
#include "UnitTest++.h"
#include "rrc_api.h"
#include "rrUtils.h"
#include "rrException.h"
#include "src/rrSBMLTestSuiteSimulation_CAPI.h"

using namespace UnitTest;
using namespace std;
using namespace rr;

extern string     gTSModelsPath;
extern string   gTempFolder;
extern bool        gDebug;

SUITE(SBML_TEST_SUITE)
{
 TEST(1) {CHECK(RunTest("l2v4", 1)); }
 TEST(2) { CHECK(RunTest("l2v4", 2)); }
 TEST(3) { CHECK(RunTest("l2v4", 3)); }
 TEST(4) { CHECK(RunTest("l2v4", 4)); }
 TEST(5) { CHECK(RunTest("l2v4", 5)); }
 TEST(6) { CHECK(RunTest("l2v4", 6)); }
 TEST(7) { CHECK(RunTest("l2v4", 7)); }
 TEST(8) { CHECK(RunTest("l2v4", 8)); }
 TEST(9) { CHECK(RunTest("l2v4", 9)); }
 TEST(10) { CHECK(RunTest("l2v4", 10)); }
 TEST(11) { CHECK(RunTest("l2v4", 11)); }
 TEST(12) { CHECK(RunTest("l2v4", 12)); }
 TEST(13) { CHECK(RunTest("l2v4", 13)); }
 TEST(14) { CHECK(RunTest("l2v4", 14)); }
 TEST(15) { CHECK(RunTest("l2v4", 15)); }
 TEST(16) { CHECK(RunTest("l2v4", 16)); }
 TEST(17) { CHECK(RunTest("l2v4", 17)); }
 TEST(18) { CHECK(RunTest("l2v4", 18)); }
 TEST(19) { CHECK(RunTest("l2v4", 19)); }
 TEST(20) { CHECK(RunTest("l2v4", 20)); }
 TEST(21) { CHECK(RunTest("l2v4", 21)); }
 TEST(22) { CHECK(RunTest("l2v4", 22)); }
 TEST(23) { CHECK(RunTest("l2v4", 23)); }
 TEST(24) { CHECK(RunTest("l2v4", 24)); }
 TEST(25) { CHECK(RunTest("l2v4", 25)); }
 TEST(26) { CHECK(RunTest("l2v4", 26)); }
 TEST(27) { CHECK(RunTest("l2v4", 27)); }
 TEST(28) { CHECK(RunTest("l2v4", 28)); }
 TEST(29) { CHECK(RunTest("l2v4", 29)); }
 TEST(30) { CHECK(RunTest("l2v4", 30)); }
 TEST(31) { CHECK(RunTest("l2v4", 31)); }
 TEST(32) { CHECK(RunTest("l2v4", 32)); }
 TEST(33) { CHECK(RunTest("l2v4", 33)); }

 TEST(34) { CHECK(RunTest("l2v4", 34)); }
 TEST(35) { CHECK(RunTest("l2v4", 35)); }
 TEST(36) { CHECK(RunTest("l2v4", 36)); }
 TEST(37) { CHECK(RunTest("l2v4", 37)); }
 TEST(38) { CHECK(RunTest("l2v4", 38)); }
 //TEST(39) { CHECK(RunTest("l2v4", 39)); } // AlgebraicRule
 //TEST(40) { CHECK(RunTest("l2v4", 40)); } // AlgebraicRule
 TEST(41) { CHECK(RunTest("l2v4", 41)); }
 TEST(42) { CHECK(RunTest("l2v4", 42)); }
 TEST(43) { CHECK(RunTest("l2v4", 43)); }
 TEST(44) { CHECK(RunTest("l2v4", 44)); }
 TEST(45) { CHECK(RunTest("l2v4", 45)); }
 TEST(46) { CHECK(RunTest("l2v4", 46)); }
 TEST(47) { CHECK(RunTest("l2v4", 47)); }
 TEST(48) { CHECK(RunTest("l2v4", 48)); }
 TEST(49) { CHECK(RunTest("l2v4", 49)); }
 TEST(50) { CHECK(RunTest("l2v4", 50)); }
 TEST(51) { CHECK(RunTest("l2v4", 51)); }
 TEST(52) { CHECK(RunTest("l2v4", 52)); }
 TEST(53) { CHECK(RunTest("l2v4", 53)); }
 TEST(54) { CHECK(RunTest("l2v4", 54)); }
 TEST(55) { CHECK(RunTest("l2v4", 55)); }
 TEST(56) { CHECK(RunTest("l2v4", 56)); }
 TEST(57) { CHECK(RunTest("l2v4", 57)); }
 TEST(58) { CHECK(RunTest("l2v4", 58)); }
 TEST(59) { CHECK(RunTest("l2v4", 59)); }
 TEST(60) { CHECK(RunTest("l2v4", 60)); }
 TEST(61) { CHECK(RunTest("l2v4", 61)); }
 TEST(62) { CHECK(RunTest("l2v4", 62)); }
 TEST(63) { CHECK(RunTest("l2v4", 63)); }
 TEST(64) { CHECK(RunTest("l2v4", 64)); }
 TEST(65) { CHECK(RunTest("l2v4", 65)); }
 TEST(66) { CHECK(RunTest("l2v4", 66)); }
 TEST(67) { CHECK(RunTest("l2v4", 67)); }
 TEST(68) { CHECK(RunTest("l2v4", 68)); }
 TEST(69) { CHECK(RunTest("l2v4", 69)); }
 TEST(70) { CHECK(RunTest("l2v4", 70)); }
 TEST(71) { CHECK(RunTest("l2v4", 71)); }
 TEST(72) { CHECK(RunTest("l2v4", 72)); }
 TEST(73) { CHECK(RunTest("l2v4", 73)); }
 TEST(74) { CHECK(RunTest("l2v4", 74)); }
 TEST(75) { CHECK(RunTest("l2v4", 75)); }
 TEST(76) { CHECK(RunTest("l2v4", 76)); }
 TEST(77) { CHECK(RunTest("l2v4", 77)); }
 TEST(78) { CHECK(RunTest("l2v4", 78)); }
 TEST(79) { CHECK(RunTest("l2v4", 79)); }
 TEST(80) { CHECK(RunTest("l2v4", 80)); }
 TEST(81) { CHECK(RunTest("l2v4", 81)); }
 TEST(82) { CHECK(RunTest("l2v4", 82)); }
 TEST(83) { CHECK(RunTest("l2v4", 83)); }
 TEST(84) { CHECK(RunTest("l2v4", 84)); }
 TEST(85) { CHECK(RunTest("l2v4", 85)); }
 TEST(86) { CHECK(RunTest("l2v4", 86)); }
 TEST(87) { CHECK(RunTest("l2v4", 87)); }
 TEST(88) { CHECK(RunTest("l2v4", 88)); }
 TEST(89) { CHECK(RunTest("l2v4", 89)); }
 TEST(90) { CHECK(RunTest("l2v4", 90)); }
 TEST(91) { CHECK(RunTest("l2v4", 91)); }
 TEST(92) { CHECK(RunTest("l2v4", 92)); }
 TEST(93) { CHECK(RunTest("l2v4", 93)); }
 TEST(94) { CHECK(RunTest("l2v4", 94)); }
 TEST(95) { CHECK(RunTest("l2v4", 95)); }
 TEST(96) { CHECK(RunTest("l2v4", 96)); }
 TEST(97) { CHECK(RunTest("l2v4", 97)); }
 TEST(98) { CHECK(RunTest("l2v4", 98)); }
 TEST(99) { CHECK(RunTest("l2v4", 99)); }
 TEST(100) { CHECK(RunTest("l2v4", 100)); }
 TEST(101) { CHECK(RunTest("l2v4", 101)); }
 TEST(102) { CHECK(RunTest("l2v4", 102)); }
 TEST(103) { CHECK(RunTest("l2v4", 103)); }
 TEST(104) { CHECK(RunTest("l2v4", 104)); }
 TEST(105) { CHECK(RunTest("l2v4", 105)); }
 TEST(106) { CHECK(RunTest("l2v4", 106)); }
 TEST(107) { CHECK(RunTest("l2v4", 107)); }
 TEST(108) { CHECK(RunTest("l2v4", 108)); }
 TEST(109) { CHECK(RunTest("l2v4", 109)); }
 TEST(110) { CHECK(RunTest("l2v4", 110)); }
 TEST(111) { CHECK(RunTest("l2v4", 111)); }
 TEST(112) { CHECK(RunTest("l2v4", 112)); }
 TEST(113) { CHECK(RunTest("l2v4", 113)); }
 TEST(114) { CHECK(RunTest("l2v4", 114)); }
 TEST(115) { CHECK(RunTest("l2v4", 115)); }
 TEST(116) { CHECK(RunTest("l2v4", 116)); }
 TEST(117) { CHECK(RunTest("l2v4", 117)); }
 TEST(118) { CHECK(RunTest("l2v4", 118)); }
 TEST(119) { CHECK(RunTest("l2v4", 119)); }
 TEST(120) { CHECK(RunTest("l2v4", 120)); }
 TEST(121) { CHECK(RunTest("l2v4", 121)); }
 TEST(122) { CHECK(RunTest("l2v4", 122)); }
 TEST(123) { CHECK(RunTest("l2v4", 123)); }
 TEST(124) { CHECK(RunTest("l2v4", 124)); }
 TEST(125) { CHECK(RunTest("l2v4", 125)); }
 TEST(126) { CHECK(RunTest("l2v4", 126)); }
 TEST(127) { CHECK(RunTest("l2v4", 127)); }
 TEST(128) { CHECK(RunTest("l2v4", 128)); }
 TEST(129) { CHECK(RunTest("l2v4", 129)); }
 TEST(130) { CHECK(RunTest("l2v4", 130)); }
 TEST(131) { CHECK(RunTest("l2v4", 131)); }
 TEST(132) { CHECK(RunTest("l2v4", 132)); }
 TEST(133) { CHECK(RunTest("l2v4", 133)); }
 TEST(134) { CHECK(RunTest("l2v4", 134)); }
 TEST(135) { CHECK(RunTest("l2v4", 135)); }
 TEST(136) { CHECK(RunTest("l2v4", 136)); }
 TEST(137) { CHECK(RunTest("l2v4", 137)); }
 TEST(138) { CHECK(RunTest("l2v4", 138)); }
 TEST(139) { CHECK(RunTest("l2v4", 139)); }
 TEST(140) { CHECK(RunTest("l2v4", 140)); }
 TEST(141) { CHECK(RunTest("l2v4", 141)); }
 TEST(142) { CHECK(RunTest("l2v4", 142)); }
 TEST(143) { CHECK(RunTest("l2v4", 143)); }
 TEST(144) { CHECK(RunTest("l2v4", 144)); }
 TEST(145) { CHECK(RunTest("l2v4", 145)); }
 TEST(146) { CHECK(RunTest("l2v4", 146)); }
 TEST(147) { CHECK(RunTest("l2v4", 147)); }
 TEST(148) { CHECK(RunTest("l2v4", 148)); }
 TEST(149) { CHECK(RunTest("l2v4", 149)); }
 TEST(150) { CHECK(RunTest("l2v4", 150)); }
 TEST(151) { CHECK(RunTest("l2v4", 151)); }
 TEST(152) { CHECK(RunTest("l2v4", 152)); }
 TEST(153) { CHECK(RunTest("l2v4", 153)); }
 TEST(154) { CHECK(RunTest("l2v4", 154)); }
 TEST(155) { CHECK(RunTest("l2v4", 155)); }
 TEST(156) { CHECK(RunTest("l2v4", 156)); }
 TEST(157) { CHECK(RunTest("l2v4", 157)); }
 TEST(158) { CHECK(RunTest("l2v4", 158)); }
 TEST(159) { CHECK(RunTest("l2v4", 159)); }
 TEST(160) { CHECK(RunTest("l2v4", 160)); }
 TEST(161) { CHECK(RunTest("l2v4", 161)); }
 TEST(162) { CHECK(RunTest("l2v4", 162)); }
 TEST(163) { CHECK(RunTest("l2v4", 163)); }
 TEST(164) { CHECK(RunTest("l2v4", 164)); }
 TEST(165) { CHECK(RunTest("l2v4", 165)); }
 TEST(166) { CHECK(RunTest("l2v4", 166)); }
 TEST(167) { CHECK(RunTest("l2v4", 167)); }
 TEST(168) { CHECK(RunTest("l2v4", 168)); }
 TEST(169) { CHECK(RunTest("l2v4", 169)); }
 TEST(170) { CHECK(RunTest("l2v4", 170)); }
 TEST(171) { CHECK(RunTest("l2v4", 171)); }
 TEST(172) { CHECK(RunTest("l2v4", 172)); }
 TEST(173) { CHECK(RunTest("l2v4", 173)); }
 TEST(174) { CHECK(RunTest("l2v4", 174)); }
 TEST(175) { CHECK(RunTest("l2v4", 175)); }
 TEST(176) { CHECK(RunTest("l2v4", 176)); }
 TEST(177) { CHECK(RunTest("l2v4", 177)); }
 TEST(178) { CHECK(RunTest("l2v4", 178)); }
 TEST(179) { CHECK(RunTest("l2v4", 179)); }
 TEST(180) { CHECK(RunTest("l2v4", 180)); }
 TEST(181) { CHECK(RunTest("l2v4", 181)); }
 //TEST(182) { CHECK(RunTest("l2v4", 182)); } // AlgebraicRule
 TEST(183) { CHECK(RunTest("l2v4", 183)); }
 //TEST(184) { CHECK(RunTest("l2v4", 184)); } // AlgebraicRule
 TEST(185) { CHECK(RunTest("l2v4", 185)); }
 TEST(186) { CHECK(RunTest("l2v4", 186)); }
 TEST(187) { CHECK(RunTest("l2v4", 187)); }
 TEST(188) { CHECK(RunTest("l2v4", 188)); }
 TEST(189) { CHECK(RunTest("l2v4", 189)); }
 TEST(190) { CHECK(RunTest("l2v4", 190)); }
 TEST(191) { CHECK(RunTest("l2v4", 191)); }
 TEST(192) { CHECK(RunTest("l2v4", 192)); }
 TEST(193) { CHECK(RunTest("l2v4", 193)); }
 TEST(194) { CHECK(RunTest("l2v4", 194)); }
 TEST(195) { CHECK(RunTest("l2v4", 195)); }
 TEST(196) { CHECK(RunTest("l2v4", 196)); }
 TEST(197) { CHECK(RunTest("l2v4", 197)); }
 TEST(198) { CHECK(RunTest("l2v4", 198)); }
 TEST(199) { CHECK(RunTest("l2v4", 199)); }
 TEST(200) { CHECK(RunTest("l2v4", 200)); }
 TEST(201) { CHECK(RunTest("l2v4", 201)); }
 TEST(202) { CHECK(RunTest("l2v4", 202)); }
 TEST(203) { CHECK(RunTest("l2v4", 203)); }
 TEST(204) { CHECK(RunTest("l2v4", 204)); }
 TEST(205) { CHECK(RunTest("l2v4", 205)); }
 TEST(206) { CHECK(RunTest("l2v4", 206)); }
 TEST(207) { CHECK(RunTest("l2v4", 207)); }
 TEST(208) { CHECK(RunTest("l2v4", 208)); }
 TEST(209) { CHECK(RunTest("l2v4", 209)); }
 TEST(210) { CHECK(RunTest("l2v4", 210)); }
 TEST(211) { CHECK(RunTest("l2v4", 211)); }
 TEST(212) { CHECK(RunTest("l2v4", 212)); }
 TEST(213) { CHECK(RunTest("l2v4", 213)); }
 TEST(214) { CHECK(RunTest("l2v4", 214)); }
 TEST(215) { CHECK(RunTest("l2v4", 215)); }
 TEST(216) { CHECK(RunTest("l2v4", 216)); }
 TEST(217) { CHECK(RunTest("l2v4", 217)); }
 TEST(218) { CHECK(RunTest("l2v4", 218)); }
 TEST(219) { CHECK(RunTest("l2v4", 219)); }
 TEST(220) { CHECK(RunTest("l2v4", 220)); }
 TEST(221) { CHECK(RunTest("l2v4", 221)); }
 TEST(222) { CHECK(RunTest("l2v4", 222)); }
 TEST(223) { CHECK(RunTest("l2v4", 223)); }
 TEST(224) { CHECK(RunTest("l2v4", 224)); }
 TEST(225) { CHECK(RunTest("l2v4", 225)); }
 TEST(226) { CHECK(RunTest("l2v4", 226)); }
 TEST(227) { CHECK(RunTest("l2v4", 227)); }
 TEST(228) { CHECK(RunTest("l2v4", 228)); }
 TEST(229) { CHECK(RunTest("l2v4", 229)); }
 TEST(230) { CHECK(RunTest("l2v4", 230)); }
 TEST(231) { CHECK(RunTest("l2v4", 231)); }
 TEST(232) { CHECK(RunTest("l2v4", 232)); }
 TEST(233) { CHECK(RunTest("l2v4", 233)); }
 TEST(234) { CHECK(RunTest("l2v4", 234)); }
 TEST(235) { CHECK(RunTest("l2v4", 235)); }
 TEST(236) { CHECK(RunTest("l2v4", 236)); }
 TEST(237) { CHECK(RunTest("l2v4", 237)); }
 TEST(238) { CHECK(RunTest("l2v4", 238)); }
 TEST(239) { CHECK(RunTest("l2v4", 239)); }
 TEST(240) { CHECK(RunTest("l2v4", 240)); }
 TEST(241) { CHECK(RunTest("l2v4", 241)); }
 TEST(242) { CHECK(RunTest("l2v4", 242)); }
 TEST(243) { CHECK(RunTest("l2v4", 243)); }
 TEST(244) { CHECK(RunTest("l2v4", 244)); }
 TEST(245) { CHECK(RunTest("l2v4", 245)); }
 TEST(246) { CHECK(RunTest("l2v4", 246)); }
 TEST(247) { CHECK(RunTest("l2v4", 247)); }
 TEST(248) { CHECK(RunTest("l2v4", 248)); }
 TEST(249) { CHECK(RunTest("l2v4", 249)); }
 TEST(250) { CHECK(RunTest("l2v4", 250)); }
 TEST(251) { CHECK(RunTest("l2v4", 251)); }
 TEST(252) { CHECK(RunTest("l2v4", 252)); }
 TEST(253) { CHECK(RunTest("l2v4", 253)); }
 TEST(254) { CHECK(RunTest("l2v4", 254)); }
 TEST(255) { CHECK(RunTest("l2v4", 255)); }
 TEST(256) { CHECK(RunTest("l2v4", 256)); }
 TEST(257) { CHECK(RunTest("l2v4", 257)); }
 TEST(258) { CHECK(RunTest("l2v4", 258)); }
 TEST(259) { CHECK(RunTest("l2v4", 259)); }
 TEST(260) { CHECK(RunTest("l2v4", 260)); }
 TEST(261) { CHECK(RunTest("l2v4", 261)); }
 TEST(262) { CHECK(RunTest("l2v4", 262)); }
 TEST(263) { CHECK(RunTest("l2v4", 263)); }
 TEST(264) { CHECK(RunTest("l2v4", 264)); }
 TEST(265) { CHECK(RunTest("l2v4", 265)); }
 TEST(266) { CHECK(RunTest("l2v4", 266)); }
 TEST(267) { CHECK(RunTest("l2v4", 267)); }
 TEST(268) { CHECK(RunTest("l2v4", 268)); }
 TEST(269) { CHECK(RunTest("l2v4", 269)); }
 TEST(270) { CHECK(RunTest("l2v4", 270)); }
 TEST(271) { CHECK(RunTest("l2v4", 271)); }
 TEST(272) { CHECK(RunTest("l2v4", 272)); }
 TEST(273) { CHECK(RunTest("l2v4", 273)); }
 TEST(274) { CHECK(RunTest("l2v4", 274)); }
 TEST(275) { CHECK(RunTest("l2v4", 275)); }
 TEST(276) { CHECK(RunTest("l2v4", 276)); }
 TEST(277) { CHECK(RunTest("l2v4", 277)); }
 TEST(278) { CHECK(RunTest("l2v4", 278)); }
 TEST(279) { CHECK(RunTest("l2v4", 279)); }
 TEST(280) { CHECK(RunTest("l2v4", 280)); }
 TEST(281) { CHECK(RunTest("l2v4", 281)); }
 TEST(282) { CHECK(RunTest("l2v4", 282)); }
 TEST(283) { CHECK(RunTest("l2v4", 283)); }
 TEST(284) { CHECK(RunTest("l2v4", 284)); }
 TEST(285) { CHECK(RunTest("l2v4", 285)); }
 TEST(286) { CHECK(RunTest("l2v4", 286)); }
 TEST(287) { CHECK(RunTest("l2v4", 287)); }
 TEST(288) { CHECK(RunTest("l2v4", 288)); }
 TEST(289) { CHECK(RunTest("l2v4", 289)); }
 TEST(290) { CHECK(RunTest("l2v4", 290)); }
 TEST(291) { CHECK(RunTest("l2v4", 291)); }
 TEST(292) { CHECK(RunTest("l2v4", 292)); }
 TEST(293) { CHECK(RunTest("l2v4", 293)); }
 TEST(294) { CHECK(RunTest("l2v4", 294)); }
 TEST(295) { CHECK(RunTest("l2v4", 295)); }
 TEST(296) { CHECK(RunTest("l2v4", 296)); }
 TEST(297) { CHECK(RunTest("l2v4", 297)); }
 TEST(298) { CHECK(RunTest("l2v4", 298)); }
 TEST(299) { CHECK(RunTest("l2v4", 299)); }
 TEST(300) { CHECK(RunTest("l2v4", 300)); }
 TEST(301) { CHECK(RunTest("l2v4", 301)); }
 TEST(302) { CHECK(RunTest("l2v4", 302)); }
 TEST(303) { CHECK(RunTest("l2v4", 303)); }
 TEST(304) { CHECK(RunTest("l2v4", 304)); }
 TEST(305) { CHECK(RunTest("l2v4", 305)); }
 TEST(306) { CHECK(RunTest("l2v4", 306)); }
 TEST(307) { CHECK(RunTest("l2v4", 307)); }
 TEST(308) { CHECK(RunTest("l2v4", 308)); }
 TEST(309) { CHECK(RunTest("l2v4", 309)); }
 TEST(310) { CHECK(RunTest("l2v4", 310)); }
 TEST(311) { CHECK(RunTest("l2v4", 311)); }
 TEST(312) { CHECK(RunTest("l2v4", 312)); }
 TEST(313) { CHECK(RunTest("l2v4", 313)); }
 TEST(314) { CHECK(RunTest("l2v4", 314)); }
 TEST(315) { CHECK(RunTest("l2v4", 315)); }
 TEST(316) { CHECK(RunTest("l2v4", 316)); }
 TEST(317) { CHECK(RunTest("l2v4", 317)); }
 TEST(318) { CHECK(RunTest("l2v4", 318)); }
 TEST(319) { CHECK(RunTest("l2v4", 319)); }
 TEST(320) { CHECK(RunTest("l2v4", 320)); }
 TEST(321) { CHECK(RunTest("l2v4", 321)); }
 TEST(322) { CHECK(RunTest("l2v4", 322)); }
 TEST(323) { CHECK(RunTest("l2v4", 323)); }
 TEST(324) { CHECK(RunTest("l2v4", 324)); }
 TEST(325) { CHECK(RunTest("l2v4", 325)); }
 TEST(326) { CHECK(RunTest("l2v4", 326)); }
 TEST(327) { CHECK(RunTest("l2v4", 327)); }
 TEST(328) { CHECK(RunTest("l2v4", 328)); }
 TEST(329) { CHECK(RunTest("l2v4", 329)); }
 TEST(330) { CHECK(RunTest("l2v4", 330)); }
 TEST(331) { CHECK(RunTest("l2v4", 331)); }
 TEST(332) { CHECK(RunTest("l2v4", 332)); }
 TEST(333) { CHECK(RunTest("l2v4", 333)); }
 TEST(334) { CHECK(RunTest("l2v4", 334)); }
 TEST(335) { CHECK(RunTest("l2v4", 335)); }
 TEST(336) { CHECK(RunTest("l2v4", 336)); }
 TEST(337) { CHECK(RunTest("l2v4", 337)); }
 TEST(338) { CHECK(RunTest("l2v4", 338)); }
 TEST(339) { CHECK(RunTest("l2v4", 339)); }
 TEST(340) { CHECK(RunTest("l2v4", 340)); }
 TEST(341) { CHECK(RunTest("l2v4", 341)); }
 TEST(342) { CHECK(RunTest("l2v4", 342)); }
 TEST(343) { CHECK(RunTest("l2v4", 343)); }
 TEST(344) { CHECK(RunTest("l2v4", 344)); }
 TEST(345) { CHECK(RunTest("l2v4", 345)); }
 TEST(346) { CHECK(RunTest("l2v4", 346)); }
 TEST(347) { CHECK(RunTest("l2v4", 347)); }
 TEST(348) { CHECK(RunTest("l2v4", 348)); }
 TEST(349) { CHECK(RunTest("l2v4", 349)); }
 TEST(350) { CHECK(RunTest("l2v4", 350)); }
 TEST(351) { CHECK(RunTest("l2v4", 351)); }
 TEST(352) { CHECK(RunTest("l2v4", 352)); }
 TEST(353) { CHECK(RunTest("l2v4", 353)); }
 TEST(354) { CHECK(RunTest("l2v4", 354)); }
 TEST(355) { CHECK(RunTest("l2v4", 355)); }
 TEST(356) { CHECK(RunTest("l2v4", 356)); }
 TEST(357) { CHECK(RunTest("l2v4", 357)); }
 TEST(358) { CHECK(RunTest("l2v4", 358)); }
 TEST(359) { CHECK(RunTest("l2v4", 359)); }
 TEST(360) { CHECK(RunTest("l2v4", 360)); }
 TEST(361) { CHECK(RunTest("l2v4", 361)); }
 TEST(362) { CHECK(RunTest("l2v4", 362)); }
 TEST(363) { CHECK(RunTest("l2v4", 363)); }
 TEST(364) { CHECK(RunTest("l2v4", 364)); }
 TEST(365) { CHECK(RunTest("l2v4", 365)); }
 TEST(366) { CHECK(RunTest("l2v4", 366)); }
 TEST(367) { CHECK(RunTest("l2v4", 367)); }
 TEST(368) { CHECK(RunTest("l2v4", 368)); }
 TEST(369) { CHECK(RunTest("l2v4", 369)); }
 TEST(370) { CHECK(RunTest("l2v4", 370)); }
 TEST(371) { CHECK(RunTest("l2v4", 371)); }
 TEST(372) { CHECK(RunTest("l2v4", 372)); }
 TEST(373) { CHECK(RunTest("l2v4", 373)); }
 TEST(374) { CHECK(RunTest("l2v4", 374)); }
 TEST(375) { CHECK(RunTest("l2v4", 375)); }
 TEST(376) { CHECK(RunTest("l2v4", 376)); }
 TEST(377) { CHECK(RunTest("l2v4", 377)); }
 TEST(378) { CHECK(RunTest("l2v4", 378)); }
 TEST(379) { CHECK(RunTest("l2v4", 379)); }
 TEST(380) { CHECK(RunTest("l2v4", 380)); }
 TEST(381) { CHECK(RunTest("l2v4", 381)); }
 TEST(382) { CHECK(RunTest("l2v4", 382)); }
 TEST(383) { CHECK(RunTest("l2v4", 383)); }
 TEST(384) { CHECK(RunTest("l2v4", 384)); }
 TEST(385) { CHECK(RunTest("l2v4", 385)); }
 TEST(386) { CHECK(RunTest("l2v4", 386)); }
 TEST(387) { CHECK(RunTest("l2v4", 387)); }
 TEST(388) { CHECK(RunTest("l2v4", 388)); }
 TEST(389) { CHECK(RunTest("l2v4", 389)); }
 TEST(390) { CHECK(RunTest("l2v4", 390)); }
 TEST(391) { CHECK(RunTest("l2v4", 391)); }
 TEST(392) { CHECK(RunTest("l2v4", 392)); }
 TEST(393) { CHECK(RunTest("l2v4", 393)); }
 TEST(394) { CHECK(RunTest("l2v4", 394)); }
 TEST(395) { CHECK(RunTest("l2v4", 395)); }
 TEST(396) { CHECK(RunTest("l2v4", 396)); }
 TEST(397) { CHECK(RunTest("l2v4", 397)); }
 TEST(398) { CHECK(RunTest("l2v4", 398)); }
 TEST(399) { CHECK(RunTest("l2v4", 399)); }
 TEST(400) { CHECK(RunTest("l2v4", 400)); }
 TEST(401) { CHECK(RunTest("l2v4", 401)); }
 TEST(402) { CHECK(RunTest("l2v4", 402)); }
 TEST(403) { CHECK(RunTest("l2v4", 403)); }
 TEST(404) { CHECK(RunTest("l2v4", 404)); }
 TEST(405) { CHECK(RunTest("l2v4", 405)); }
 TEST(406) { CHECK(RunTest("l2v4", 406)); }
 TEST(407) { CHECK(RunTest("l2v4", 407)); }
 TEST(408) { CHECK(RunTest("l2v4", 408)); }
 TEST(409) { CHECK(RunTest("l2v4", 409)); }
 TEST(410) { CHECK(RunTest("l2v4", 410)); }
 TEST(411) { CHECK(RunTest("l2v4", 411)); }
 TEST(412) { CHECK(RunTest("l2v4", 412)); }
 TEST(413) { CHECK(RunTest("l2v4", 413)); }
 TEST(414) { CHECK(RunTest("l2v4", 414)); }
 TEST(415) { CHECK(RunTest("l2v4", 415)); }
 TEST(416) { CHECK(RunTest("l2v4", 416)); }
 TEST(417) { CHECK(RunTest("l2v4", 417)); }
 TEST(418) { CHECK(RunTest("l2v4", 418)); }
 TEST(419) { CHECK(RunTest("l2v4", 419)); }
 TEST(420) { CHECK(RunTest("l2v4", 420)); }
 TEST(421) { CHECK(RunTest("l2v4", 421)); }
 TEST(422) { CHECK(RunTest("l2v4", 422)); }
 TEST(423) { CHECK(RunTest("l2v4", 423)); }
 TEST(424) { CHECK(RunTest("l2v4", 424)); }
 TEST(425) { CHECK(RunTest("l2v4", 425)); }
 TEST(426) { CHECK(RunTest("l2v4", 426)); }
 TEST(427) { CHECK(RunTest("l2v4", 427)); }
 TEST(428) { CHECK(RunTest("l2v4", 428)); }
 TEST(429) { CHECK(RunTest("l2v4", 429)); }
 TEST(430) { CHECK(RunTest("l2v4", 430)); }
 TEST(431) { CHECK(RunTest("l2v4", 431)); }
 TEST(432) { CHECK(RunTest("l2v4", 432)); }
 TEST(433) { CHECK(RunTest("l2v4", 433)); }
 TEST(434) { CHECK(RunTest("l2v4", 434)); }
 TEST(435) { CHECK(RunTest("l2v4", 435)); }
 TEST(436) { CHECK(RunTest("l2v4", 436)); }
 TEST(437) { CHECK(RunTest("l2v4", 437)); }
 TEST(438) { CHECK(RunTest("l2v4", 438)); }
 TEST(439) { CHECK(RunTest("l2v4", 439)); }
 TEST(440) { CHECK(RunTest("l2v4", 440)); }
 TEST(441) { CHECK(RunTest("l2v4", 441)); }
 TEST(442) { CHECK(RunTest("l2v4", 442)); }
 TEST(443) { CHECK(RunTest("l2v4", 443)); }
 TEST(444) { CHECK(RunTest("l2v4", 444)); }
 TEST(445) { CHECK(RunTest("l2v4", 445)); }
 TEST(446) { CHECK(RunTest("l2v4", 446)); }
 TEST(447) { CHECK(RunTest("l2v4", 447)); }
 TEST(448) { CHECK(RunTest("l2v4", 448)); }
 TEST(449) { CHECK(RunTest("l2v4", 449)); }
 TEST(450) { CHECK(RunTest("l2v4", 450)); }
 TEST(451) { CHECK(RunTest("l2v4", 451)); }
 TEST(452) { CHECK(RunTest("l2v4", 452)); }
 TEST(453) { CHECK(RunTest("l2v4", 453)); }
 TEST(454) { CHECK(RunTest("l2v4", 454)); }
 TEST(455) { CHECK(RunTest("l2v4", 455)); }
 TEST(456) { CHECK(RunTest("l2v4", 456)); }
 TEST(457) { CHECK(RunTest("l2v4", 457)); }
 TEST(458) { CHECK(RunTest("l2v4", 458)); }
 TEST(459) { CHECK(RunTest("l2v4", 459)); }
 TEST(460) { CHECK(RunTest("l2v4", 460)); }
 TEST(461) { CHECK(RunTest("l2v4", 461)); }
 TEST(462) { CHECK(RunTest("l2v4", 462)); }
 TEST(463) { CHECK(RunTest("l2v4", 463)); }
 TEST(464) { CHECK(RunTest("l2v4", 464)); }
 TEST(465) { CHECK(RunTest("l2v4", 465)); }
 TEST(466) { CHECK(RunTest("l2v4", 466)); }
 TEST(467) { CHECK(RunTest("l2v4", 467)); }
 TEST(468) { CHECK(RunTest("l2v4", 468)); }
 TEST(469) { CHECK(RunTest("l2v4", 469)); }
 TEST(470) { CHECK(RunTest("l2v4", 470)); }
 TEST(471) { CHECK(RunTest("l2v4", 471)); }
 TEST(472) { CHECK(RunTest("l2v4", 472)); }
 TEST(473) { CHECK(RunTest("l2v4", 473)); }
 TEST(474) { CHECK(RunTest("l2v4", 474)); }
 TEST(475) { CHECK(RunTest("l2v4", 475)); }
 TEST(476) { CHECK(RunTest("l2v4", 476)); }
 TEST(477) { CHECK(RunTest("l2v4", 477)); }
 TEST(478) { CHECK(RunTest("l2v4", 478)); }
 TEST(479) { CHECK(RunTest("l2v4", 479)); }
 TEST(480) { CHECK(RunTest("l2v4", 480)); }
 TEST(481) { CHECK(RunTest("l2v4", 481)); }
 TEST(482) { CHECK(RunTest("l2v4", 482)); }
 TEST(483) { CHECK(RunTest("l2v4", 483)); }
 TEST(484) { CHECK(RunTest("l2v4", 484)); }
 TEST(485) { CHECK(RunTest("l2v4", 485)); }
 TEST(486) { CHECK(RunTest("l2v4", 486)); }
 TEST(487) { CHECK(RunTest("l2v4", 487)); }
 TEST(488) { CHECK(RunTest("l2v4", 488)); }
 TEST(489) { CHECK(RunTest("l2v4", 489)); }
 TEST(490) { CHECK(RunTest("l2v4", 490)); }
 TEST(491) { CHECK(RunTest("l2v4", 491)); }
 TEST(492) { CHECK(RunTest("l2v4", 492)); }
 TEST(493) { CHECK(RunTest("l2v4", 493)); }
 TEST(494) { CHECK(RunTest("l2v4", 494)); }
 TEST(495) { CHECK(RunTest("l2v4", 495)); }
 TEST(496) { CHECK(RunTest("l2v4", 496)); }
 TEST(497) { CHECK(RunTest("l2v4", 497)); }
 TEST(498) { CHECK(RunTest("l2v4", 498)); }
 TEST(499) { CHECK(RunTest("l2v4", 499)); }
 TEST(500) { CHECK(RunTest("l2v4", 500)); }
 TEST(501) { CHECK(RunTest("l2v4", 501)); }
 TEST(502) { CHECK(RunTest("l2v4", 502)); }
 TEST(503) { CHECK(RunTest("l2v4", 503)); }
 TEST(504) { CHECK(RunTest("l2v4", 504)); }
 TEST(505) { CHECK(RunTest("l2v4", 505)); }
 TEST(506) { CHECK(RunTest("l2v4", 506)); }
 TEST(507) { CHECK(RunTest("l2v4", 507)); }
 TEST(508) { CHECK(RunTest("l2v4", 508)); }
 TEST(509) { CHECK(RunTest("l2v4", 509)); }
 TEST(510) { CHECK(RunTest("l2v4", 510)); }
 TEST(511) { CHECK(RunTest("l2v4", 511)); }
 TEST(512) { CHECK(RunTest("l2v4", 512)); }
 TEST(513) { CHECK(RunTest("l2v4", 513)); }
 TEST(514) { CHECK(RunTest("l2v4", 514)); }
 TEST(515) { CHECK(RunTest("l2v4", 515)); }
 TEST(516) { CHECK(RunTest("l2v4", 516)); }
 TEST(517) { CHECK(RunTest("l2v4", 517)); }
 TEST(518) { CHECK(RunTest("l2v4", 518)); }
 TEST(519) { CHECK(RunTest("l2v4", 519)); }
 TEST(520) { CHECK(RunTest("l2v4", 520)); }
 TEST(521) { CHECK(RunTest("l2v4", 521)); }
 TEST(522) { CHECK(RunTest("l2v4", 522)); }
 TEST(523) { CHECK(RunTest("l2v4", 523)); }
 TEST(524) { CHECK(RunTest("l2v4", 524)); }
 TEST(525) { CHECK(RunTest("l2v4", 525)); }
 TEST(526) { CHECK(RunTest("l2v4", 526)); }
 TEST(527) { CHECK(RunTest("l2v4", 527)); }
 TEST(528) { CHECK(RunTest("l2v4", 528)); }
 TEST(529) { CHECK(RunTest("l2v4", 529)); }
 TEST(530) { CHECK(RunTest("l2v4", 530)); }
 //TEST(531) { CHECK(RunTest("l2v4", 531)); } // AlgebraicRule
 TEST(532) { CHECK(RunTest("l2v4", 532)); }
 //TEST(533) { CHECK(RunTest("l2v4", 533)); } // AlgebraicRule
 //TEST(534) { CHECK(RunTest("l2v4", 534)); } // AlgebraicRule
 //TEST(535) { CHECK(RunTest("l2v4", 535)); } // AlgebraicRule
 //TEST(536) { CHECK(RunTest("l2v4", 536)); } // AlgebraicRule
 //TEST(537) { CHECK(RunTest("l2v4", 537)); } // AlgebraicRule
 //TEST(538) { CHECK(RunTest("l2v4", 538)); } // AlgebraicRule
 TEST(539) { CHECK(RunTest("l2v4", 539)); }
 TEST(540) { CHECK(RunTest("l2v4", 540)); }
 TEST(541) { CHECK(RunTest("l2v4", 541)); }
 TEST(542) { CHECK(RunTest("l2v4", 542)); }
 //TEST(543) { CHECK(RunTest("l2v4", 543)); } // AlgebraicRule
 TEST(544) { CHECK(RunTest("l2v4", 544)); }
 TEST(545) { CHECK(RunTest("l2v4", 545)); }
 //TEST(546) { CHECK(RunTest("l2v4", 546)); } // AlgebraicRule
 TEST(547) { CHECK(RunTest("l2v4", 547)); }
 //TEST(548) { CHECK(RunTest("l2v4", 548)); } // AlgebraicRule
 //TEST(549) { CHECK(RunTest("l2v4", 549)); } // AlgebraicRule
 //TEST(550) { CHECK(RunTest("l2v4", 550)); } // AlgebraicRule
 //TEST(551) { CHECK(RunTest("l2v4", 551)); } // AlgebraicRule
 //TEST(552) { CHECK(RunTest("l2v4", 552)); } // AlgebraicRule
 //TEST(553) { CHECK(RunTest("l2v4", 553)); } // AlgebraicRule
 //TEST(554) { CHECK(RunTest("l2v4", 554)); } // AlgebraicRule
 //TEST(555) { CHECK(RunTest("l2v4", 555)); } // AlgebraicRule
 //TEST(556) { CHECK(RunTest("l2v4", 556)); } // AlgebraicRule
 //TEST(557) { CHECK(RunTest("l2v4", 557)); } // AlgebraicRule
 //TEST(558) { CHECK(RunTest("l2v4", 558)); } // AlgebraicRule
 //TEST(559) { CHECK(RunTest("l2v4", 559)); } // AlgebraicRule
 //TEST(560) { CHECK(RunTest("l2v4", 560)); } // AlgebraicRule
 //TEST(561) { CHECK(RunTest("l2v4", 561)); } // AlgebraicRule
 //TEST(562) { CHECK(RunTest("l2v4", 562)); } // AlgebraicRule
 //TEST(563) { CHECK(RunTest("l2v4", 563)); } // AlgebraicRule
 //TEST(564) { CHECK(RunTest("l2v4", 564)); } // AlgebraicRule
 //TEST(565) { CHECK(RunTest("l2v4", 565)); } // AlgebraicRule
 //TEST(566) { CHECK(RunTest("l2v4", 566)); } // AlgebraicRule
 //TEST(567) { CHECK(RunTest("l2v4", 567)); } // AlgebraicRule
 TEST(568) { CHECK(RunTest("l2v4", 568)); }
 //TEST(569) { CHECK(RunTest("l2v4", 569)); } // AlgebraicRule
 //TEST(570) { CHECK(RunTest("l2v4", 570)); } // AlgebraicRule
 //TEST(571) { CHECK(RunTest("l2v4", 571)); } // AlgebraicRule
 TEST(572) { CHECK(RunTest("l2v4", 572)); }
 //TEST(573) { CHECK(RunTest("l2v4", 573)); } // AlgebraicRule
 TEST(574) { CHECK(RunTest("l2v4", 574)); }
 //TEST(575) { CHECK(RunTest("l2v4", 575)); } // AlgebraicRule
 //TEST(576) { CHECK(RunTest("l2v4", 576)); } // AlgebraicRule
 TEST(577) { CHECK(RunTest("l2v4", 577)); }
 TEST(578) { CHECK(RunTest("l2v4", 578)); }
 TEST(579) { CHECK(RunTest("l2v4", 579)); }
 TEST(580) { CHECK(RunTest("l2v4", 580)); }
 TEST(581) { CHECK(RunTest("l2v4", 581)); }
 TEST(582) { CHECK(RunTest("l2v4", 582)); }
 TEST(583) { CHECK(RunTest("l2v4", 583)); }
 TEST(584) { CHECK(RunTest("l2v4", 584)); }
 TEST(585) { CHECK(RunTest("l2v4", 585)); }
 TEST(586) { CHECK(RunTest("l2v4", 586)); }
 TEST(587) { CHECK(RunTest("l2v4", 587)); }
 TEST(588) { CHECK(RunTest("l2v4", 588)); }
 TEST(589) { CHECK(RunTest("l2v4", 589)); }
 TEST(590) { CHECK(RunTest("l2v4", 590)); }
 TEST(591) { CHECK(RunTest("l2v4", 591)); }
 TEST(592) { CHECK(RunTest("l2v4", 592)); }
 TEST(593) { CHECK(RunTest("l2v4", 593)); }
 TEST(594) { CHECK(RunTest("l2v4", 594)); }
 TEST(595) { CHECK(RunTest("l2v4", 595)); }
 TEST(596) { CHECK(RunTest("l2v4", 596)); }
 TEST(597) { CHECK(RunTest("l2v4", 597)); }
 TEST(598) { CHECK(RunTest("l2v4", 598)); }
 TEST(599) { CHECK(RunTest("l2v4", 599)); }
 TEST(600) { CHECK(RunTest("l2v4", 600)); }
 TEST(601) { CHECK(RunTest("l2v4", 601)); }
 TEST(602) { CHECK(RunTest("l2v4", 602)); }
 TEST(603) { CHECK(RunTest("l2v4", 603)); }
 TEST(604) { CHECK(RunTest("l2v4", 604)); }
 TEST(605) { CHECK(RunTest("l2v4", 605)); }
 TEST(606) { CHECK(RunTest("l2v4", 606)); }
 TEST(607) { CHECK(RunTest("l2v4", 607)); }
 TEST(608) { CHECK(RunTest("l2v4", 608)); }
 TEST(609) { CHECK(RunTest("l2v4", 609)); }
 TEST(610) { CHECK(RunTest("l2v4", 610)); }
 TEST(611) { CHECK(RunTest("l2v4", 611)); }
 TEST(612) { CHECK(RunTest("l2v4", 612)); }
 //TEST(613) { CHECK(RunTest("l2v4", 613)); } // AlgebraicRule
 //TEST(614) { CHECK(RunTest("l2v4", 614)); } // AlgebraicRule
 //TEST(615) { CHECK(RunTest("l2v4", 615)); } // AlgebraicRule
 TEST(616) { CHECK(RunTest("l2v4", 616)); }
 TEST(617) { CHECK(RunTest("l2v4", 617)); }
 TEST(618) { CHECK(RunTest("l2v4", 618)); }
 TEST(619) { CHECK(RunTest("l2v4", 619)); }
 TEST(620) { CHECK(RunTest("l2v4", 620)); }
 TEST(621) { CHECK(RunTest("l2v4", 621)); }
 TEST(622) { CHECK(RunTest("l2v4", 622)); }
 TEST(623) { CHECK(RunTest("l2v4", 623)); }
 TEST(624) { CHECK(RunTest("l2v4", 624)); }
 TEST(625) { CHECK(RunTest("l2v4", 625)); }
 TEST(626) { CHECK(RunTest("l2v4", 626)); }
 TEST(627) { CHECK(RunTest("l2v4", 627)); }
 //TEST(628) { CHECK(RunTest("l2v4", 628)); } // AlgebraicRule
 //TEST(629) { CHECK(RunTest("l2v4", 629)); } // AlgebraicRule
 //TEST(630) { CHECK(RunTest("l2v4", 630)); } // AlgebraicRule
 TEST(631) { CHECK(RunTest("l2v4", 631)); }
 TEST(632) { CHECK(RunTest("l2v4", 632)); }
 TEST(633) { CHECK(RunTest("l2v4", 633)); }
 TEST(634) { CHECK(RunTest("l2v4", 634)); }
 TEST(635) { CHECK(RunTest("l2v4", 635)); }
 TEST(636) { CHECK(RunTest("l2v4", 636)); }
 TEST(637) { CHECK(RunTest("l2v4", 637)); }
 TEST(638) { CHECK(RunTest("l2v4", 638)); }
 TEST(639) { CHECK(RunTest("l2v4", 639)); }
 TEST(640) { CHECK(RunTest("l2v4", 640)); }
 TEST(641) { CHECK(RunTest("l2v4", 641)); }
 TEST(642) { CHECK(RunTest("l2v4", 642)); }
 TEST(643) { CHECK(RunTest("l2v4", 643)); }
 TEST(644) { CHECK(RunTest("l2v4", 644)); }
 TEST(645) { CHECK(RunTest("l2v4", 645)); }
 TEST(646) { CHECK(RunTest("l2v4", 646)); }
 TEST(647) { CHECK(RunTest("l2v4", 647)); }
 TEST(648) { CHECK(RunTest("l2v4", 648)); }
 TEST(649) { CHECK(RunTest("l2v4", 649)); }
 TEST(650) { CHECK(RunTest("l2v4", 650)); }
 TEST(651) { CHECK(RunTest("l2v4", 651)); }
 TEST(652) { CHECK(RunTest("l2v4", 652)); }
 TEST(653) { CHECK(RunTest("l2v4", 653)); }
 TEST(654) { CHECK(RunTest("l2v4", 654)); }
 TEST(655) { CHECK(RunTest("l2v4", 655)); }
 TEST(656) { CHECK(RunTest("l2v4", 656)); }
 TEST(657) { CHECK(RunTest("l2v4", 657)); }
 //TEST(658) { CHECK(RunTest("l2v4", 658)); } // AlgebraicRule
 //TEST(659) { CHECK(RunTest("l2v4", 659)); } // AlgebraicRule
 //TEST(660) { CHECK(RunTest("l2v4", 660)); } // AlgebraicRule
 //TEST(661) { CHECK(RunTest("l2v4", 661)); } // AlgebraicRule
 //TEST(662) { CHECK(RunTest("l2v4", 662)); } // AlgebraicRule
 //TEST(663) { CHECK(RunTest("l2v4", 663)); } // AlgebraicRule
 //TEST(664) { CHECK(RunTest("l2v4", 664)); } // AlgebraicRule
 //TEST(665) { CHECK(RunTest("l2v4", 665)); } // AlgebraicRule
 //TEST(666) { CHECK(RunTest("l2v4", 666)); } // AlgebraicRule
 TEST(667) { CHECK(RunTest("l2v4", 667)); }
 TEST(668) { CHECK(RunTest("l2v4", 668)); }
 TEST(669) { CHECK(RunTest("l2v4", 669)); }
 TEST(670) { CHECK(RunTest("l2v4", 670)); }
 TEST(671) { CHECK(RunTest("l2v4", 671)); }
 TEST(672) { CHECK(RunTest("l2v4", 672)); }
 //TEST(673) { CHECK(RunTest("l2v4", 673)); } // AlgebraicRule
 //TEST(674) { CHECK(RunTest("l2v4", 674)); } // AlgebraicRule
 //TEST(675) { CHECK(RunTest("l2v4", 675)); } // AlgebraicRule
 TEST(676) { CHECK(RunTest("l2v4", 676)); }
 TEST(677) { CHECK(RunTest("l2v4", 677)); }
 TEST(678) { CHECK(RunTest("l2v4", 678)); }
 TEST(679) { CHECK(RunTest("l2v4", 679)); }
 TEST(680) { CHECK(RunTest("l2v4", 680)); }
 TEST(681) { CHECK(RunTest("l2v4", 681)); }
 TEST(682) { CHECK(RunTest("l2v4", 682)); }
 TEST(683) { CHECK(RunTest("l2v4", 683)); }
 TEST(684) { CHECK(RunTest("l2v4", 684)); }
 TEST(685) { CHECK(RunTest("l2v4", 685)); }
 TEST(686) { CHECK(RunTest("l2v4", 686)); }
 //TEST(687) { CHECK(RunTest("l2v4", 687)); } // AlgebraicRule
 TEST(688) { CHECK(RunTest("l2v4", 688)); }
 TEST(689) { CHECK(RunTest("l2v4", 689)); }
 TEST(690) { CHECK(RunTest("l2v4", 690)); }
 TEST(691) { CHECK(RunTest("l2v4", 691)); }
 TEST(692) { CHECK(RunTest("l2v4", 692)); }
 TEST(693) { CHECK(RunTest("l2v4", 693)); }
 TEST(694) { CHECK(RunTest("l2v4", 694)); }
 //TEST(695) { CHECK(RunTest("l2v4", 695)); } // AlgebraicRule
 //TEST(696) { CHECK(RunTest("l2v4", 696)); } // AlgebraicRule
 TEST(697) { CHECK(RunTest("l2v4", 697)); }
 TEST(698) { CHECK(RunTest("l2v4", 698)); }
 TEST(699) { CHECK(RunTest("l2v4", 699)); }
 TEST(700) { CHECK(RunTest("l2v4", 700)); }
 TEST(701) { CHECK(RunTest("l2v4", 701)); }
 TEST(702) { CHECK(RunTest("l2v4", 702)); }
 TEST(703) { CHECK(RunTest("l2v4", 703)); }
 TEST(704) { CHECK(RunTest("l2v4", 704)); }
 //TEST(705) { CHECK(RunTest("l2v4", 705)); } // AlgebraicRule
 TEST(706) { CHECK(RunTest("l2v4", 706)); }
 TEST(707) { CHECK(RunTest("l2v4", 707)); }
 TEST(708) { CHECK(RunTest("l2v4", 708)); }
 TEST(709) { CHECK(RunTest("l2v4", 709)); }
 TEST(710) { CHECK(RunTest("l2v4", 710)); }
 TEST(711) { CHECK(RunTest("l2v4", 711)); }
 TEST(712) { CHECK(RunTest("l2v4", 712)); }
 TEST(713) { CHECK(RunTest("l2v4", 713)); }
 TEST(714) { CHECK(RunTest("l2v4", 714)); }
 TEST(715) { CHECK(RunTest("l2v4", 715)); }
 TEST(716) { CHECK(RunTest("l2v4", 716)); }
 TEST(717) { CHECK(RunTest("l2v4", 717)); }
 TEST(718) { CHECK(RunTest("l2v4", 718)); }
 TEST(719) { CHECK(RunTest("l2v4", 719)); }
 TEST(720) { CHECK(RunTest("l2v4", 720)); }
 TEST(721) { CHECK(RunTest("l2v4", 721)); }
 TEST(722) { CHECK(RunTest("l2v4", 722)); }
 TEST(723) { CHECK(RunTest("l2v4", 723)); }
 TEST(724) { CHECK(RunTest("l2v4", 724)); }
 TEST(725) { CHECK(RunTest("l2v4", 725)); }
 TEST(726) { CHECK(RunTest("l2v4", 726)); }
 TEST(727) { CHECK(RunTest("l2v4", 727)); }
 TEST(728) { CHECK(RunTest("l2v4", 728)); }
 TEST(729) { CHECK(RunTest("l2v4", 729)); }
 TEST(730) { CHECK(RunTest("l2v4", 730)); }
 TEST(731) { CHECK(RunTest("l2v4", 731)); }
 TEST(732) { CHECK(RunTest("l2v4", 732)); }
 TEST(733) { CHECK(RunTest("l2v4", 733)); }
 TEST(734) { CHECK(RunTest("l2v4", 734)); }
 TEST(735) { CHECK(RunTest("l2v4", 735)); }
 TEST(736) { CHECK(RunTest("l2v4", 736)); }
 TEST(737) { CHECK(RunTest("l2v4", 737)); }
 TEST(738) { CHECK(RunTest("l2v4", 738)); }
 TEST(739) { CHECK(RunTest("l2v4", 739)); }
 TEST(740) { CHECK(RunTest("l2v4", 740)); }
 TEST(741) { CHECK(RunTest("l2v4", 741)); }
 TEST(742) { CHECK(RunTest("l2v4", 742)); }
 TEST(743) { CHECK(RunTest("l2v4", 743)); }
 TEST(744) { CHECK(RunTest("l2v4", 744)); }
 TEST(745) { CHECK(RunTest("l2v4", 745)); }
 TEST(746) { CHECK(RunTest("l2v4", 746)); }
 TEST(747) { CHECK(RunTest("l2v4", 747)); }
 TEST(748) { CHECK(RunTest("l2v4", 748)); } 
 TEST(749) { CHECK(RunTest("l2v4", 749)); }
 TEST(750) { CHECK(RunTest("l2v4", 750)); }
 TEST(751) { CHECK(RunTest("l2v4", 751)); }
 TEST(752) { CHECK(RunTest("l2v4", 752)); }
 TEST(753) { CHECK(RunTest("l2v4", 753)); }
 TEST(754) { CHECK(RunTest("l2v4", 754)); }
 TEST(755) { CHECK(RunTest("l2v4", 755)); }
 TEST(756) { CHECK(RunTest("l2v4", 756)); }
 TEST(757) { CHECK(RunTest("l2v4", 757)); }
 TEST(758) { CHECK(RunTest("l2v4", 758)); }
 TEST(759) { CHECK(RunTest("l2v4", 759)); }
 //TEST(760) { CHECK(RunTest("l2v4", 760)); } // AlgebraicRule
 //TEST(761) { CHECK(RunTest("l2v4", 761)); } // AlgebraicRule
 //TEST(762) { CHECK(RunTest("l2v4", 762)); } // AlgebraicRule
 TEST(763) { CHECK(RunTest("l2v4", 763)); }
 TEST(764) { CHECK(RunTest("l2v4", 764)); }
 TEST(765) { CHECK(RunTest("l2v4", 765)); }
 TEST(766) { CHECK(RunTest("l2v4", 766)); }
 TEST(767) { CHECK(RunTest("l2v4", 767)); }
 TEST(768) { CHECK(RunTest("l2v4", 768)); }
 TEST(769) { CHECK(RunTest("l2v4", 769)); }
 TEST(770) { CHECK(RunTest("l2v4", 770)); }
 TEST(771) { CHECK(RunTest("l2v4", 771)); }
 TEST(772) { CHECK(RunTest("l2v4", 772)); }
 TEST(773) { CHECK(RunTest("l2v4", 773)); }
 TEST(774) { CHECK(RunTest("l2v4", 774)); }
 TEST(775) { CHECK(RunTest("l2v4", 775)); }
 TEST(776) { CHECK(RunTest("l2v4", 776)); }
 //TEST(777) { CHECK(RunTest("l2v4", 777)); } // AlgebraicRule
 //TEST(778) { CHECK(RunTest("l2v4", 778)); } // AlgebraicRule
 //TEST(779) { CHECK(RunTest("l2v4", 779)); } // AlgebraicRule
 //TEST(780) { CHECK(RunTest("l2v4", 780)); } // AlgebraicRule
 TEST(781) { CHECK(RunTest("l2v4", 781)); }
 TEST(782) { CHECK(RunTest("l2v4", 782)); }
 TEST(783) { CHECK(RunTest("l2v4", 783)); }
 TEST(784) { CHECK(RunTest("l2v4", 784)); }
 TEST(785) { CHECK(RunTest("l2v4", 785)); }
 TEST(786) { CHECK(RunTest("l2v4", 786)); }
 TEST(787) { CHECK(RunTest("l2v4", 787)); }
 TEST(788) { CHECK(RunTest("l2v4", 788)); }
 TEST(789) { CHECK(RunTest("l2v4", 789)); }
 TEST(790) { CHECK(RunTest("l2v4", 790)); }
 TEST(791) { CHECK(RunTest("l2v4", 791)); }
 TEST(792) { CHECK(RunTest("l2v4", 792)); }
 TEST(793) { CHECK(RunTest("l2v4", 793)); }
 TEST(794) { CHECK(RunTest("l2v4", 794)); }
 TEST(795) { CHECK(RunTest("l2v4", 795)); }
 TEST(796) { CHECK(RunTest("l2v4", 796)); }
 TEST(797) { CHECK(RunTest("l2v4", 797)); }
 TEST(798) { CHECK(RunTest("l2v4", 798)); }
 TEST(799) { CHECK(RunTest("l2v4", 799)); }
 TEST(800) { CHECK(RunTest("l2v4", 800)); }
 TEST(801) { CHECK(RunTest("l2v4", 801)); }
 TEST(802) { CHECK(RunTest("l2v4", 802)); }
 TEST(803) { CHECK(RunTest("l2v4", 803)); }
 TEST(804) { CHECK(RunTest("l2v4", 804)); }
 TEST(805) { CHECK(RunTest("l2v4", 805)); }
 TEST(806) { CHECK(RunTest("l2v4", 806)); }
 TEST(807) { CHECK(RunTest("l2v4", 807)); }
 TEST(808) { CHECK(RunTest("l2v4", 808)); }
 TEST(809) { CHECK(RunTest("l2v4", 809)); }
 TEST(810) { CHECK(RunTest("l2v4", 810)); }
 TEST(811) { CHECK(RunTest("l2v4", 811)); }
 TEST(812) { CHECK(RunTest("l2v4", 812)); }
 TEST(813) { CHECK(RunTest("l2v4", 813)); }
 TEST(814) { CHECK(RunTest("l2v4", 814)); }
 TEST(815) { CHECK(RunTest("l2v4", 815)); }
 TEST(816) { CHECK(RunTest("l2v4", 816)); }
 TEST(817) { CHECK(RunTest("l2v4", 817)); }
 TEST(818) { CHECK(RunTest("l2v4", 818)); }
 TEST(819) { CHECK(RunTest("l2v4", 819)); }
 TEST(820) { CHECK(RunTest("l2v4", 820)); }
 TEST(821) { CHECK(RunTest("l2v4", 821)); }
 TEST(822) { CHECK(RunTest("l2v4", 822)); }
 TEST(823) { CHECK(RunTest("l2v4", 823)); }
 TEST(824) { CHECK(RunTest("l2v4", 824)); }
 TEST(825) { CHECK(RunTest("l2v4", 825)); }
 TEST(826) { CHECK(RunTest("l2v4", 826)); }
 TEST(827) { CHECK(RunTest("l2v4", 827)); }
 TEST(828) { CHECK(RunTest("l2v4", 828)); }
 TEST(829) { CHECK(RunTest("l2v4", 829)); }
 TEST(830) { CHECK(RunTest("l2v4", 830)); }
 TEST(831) { CHECK(RunTest("l2v4", 831)); }
 TEST(832) { CHECK(RunTest("l2v4", 832)); }
 TEST(833) { CHECK(RunTest("l2v4", 833)); }
 TEST(834) { CHECK(RunTest("l2v4", 834)); }
 TEST(835) { CHECK(RunTest("l2v4", 835)); }
 TEST(836) { CHECK(RunTest("l2v4", 836)); }
 TEST(837) { CHECK(RunTest("l2v4", 837)); }
 TEST(838) { CHECK(RunTest("l2v4", 838)); }
 TEST(839) { CHECK(RunTest("l2v4", 839)); }
 TEST(840) { CHECK(RunTest("l2v4", 840)); }
 TEST(841) { CHECK(RunTest("l2v4", 841)); }
 TEST(842) { CHECK(RunTest("l2v4", 842)); }
 TEST(843) { CHECK(RunTest("l2v4", 843)); }
//TEST(844) { CHECK(RunTest("l2v4", 844)); } // AlgebraicRule
TEST(845) { CHECK(RunTest("l2v4", 845)); }
TEST(846) { CHECK(RunTest("l2v4", 846)); }
TEST(847) { CHECK(RunTest("l2v4", 847)); }
TEST(848) { CHECK(RunTest("l2v4", 848)); }
TEST(849) { CHECK(RunTest("l2v4", 849)); }
TEST(850) { CHECK(RunTest("l2v4", 850)); }
TEST(851) { CHECK(RunTest("l2v4", 851)); }
TEST(852) { CHECK(RunTest("l2v4", 852)); }
TEST(853) { CHECK(RunTest("l2v4", 853)); }
TEST(854) { CHECK(RunTest("l2v4", 854)); }
TEST(855) { CHECK(RunTest("l2v4", 855)); }
TEST(856) { CHECK(RunTest("l2v4", 856)); }
TEST(857) { CHECK(RunTest("l2v4", 857)); }
TEST(858) { CHECK(RunTest("l2v4", 858)); }
TEST(859) { CHECK(RunTest("l2v4", 859)); }
TEST(860) { CHECK(RunTest("l2v4", 860)); }
TEST(861) { CHECK(RunTest("l2v4", 861)); }
TEST(862) { CHECK(RunTest("l2v4", 862)); }
TEST(863) { CHECK(RunTest("l2v4", 863)); }
TEST(864) { CHECK(RunTest("l2v4", 864)); }
TEST(865) { CHECK(RunTest("l2v4", 865)); }
TEST(866) { CHECK(RunTest("l2v4", 866)); }
TEST(867) { CHECK(RunTest("l2v4", 867)); }
TEST(868) { CHECK(RunTest("l2v4", 868)); }
TEST(869) { CHECK(RunTest("l2v4", 869)); }
//TEST(870) { CHECK(RunTest("l2v4", 870)); } // FastReaction
//TEST(871) { CHECK(RunTest("l2v4", 871)); } // FastReaction
//TEST(872) { CHECK(RunTest("l2v4", 872)); } // FastReaction
//TEST(873) { CHECK(RunTest("l2v4", 873)); } // FastReaction
//TEST(874) { CHECK(RunTest("l2v4", 874)); } // FastReaction
//TEST(875) { CHECK(RunTest("l2v4", 875)); } // FastReaction
//TEST(876) { CHECK(RunTest("l2v4", 876)); } // AlgebraicRule
TEST(877) { CHECK(RunTest("l2v4", 877)); }
TEST(878) { CHECK(RunTest("l2v4", 878)); }
TEST(879) { CHECK(RunTest("l2v4", 879)); }
TEST(880) { CHECK(RunTest("l2v4", 880)); }
TEST(881) { CHECK(RunTest("l2v4", 881)); }
TEST(882) { CHECK(RunTest("l2v4", 882)); }
TEST(883) { CHECK(RunTest("l2v4", 883)); }
TEST(884) { CHECK(RunTest("l2v4", 884)); }
TEST(885) { CHECK(RunTest("l2v4", 885)); }
TEST(886) { CHECK(RunTest("l2v4", 886)); }
TEST(887) { CHECK(RunTest("l2v4", 887)); }
TEST(888) { CHECK(RunTest("l2v4", 888)); }
TEST(889) { CHECK(RunTest("l2v4", 889)); }
TEST(890) { CHECK(RunTest("l2v4", 890)); }
TEST(891) { CHECK(RunTest("l2v4", 891)); }
TEST(892) { CHECK(RunTest("l2v4", 892)); }
TEST(893) { CHECK(RunTest("l2v4", 893)); }
TEST(894) { CHECK(RunTest("l2v4", 894)); }
TEST(895) { CHECK(RunTest("l2v4", 895)); }
TEST(896) { CHECK(RunTest("l2v4", 896)); }
TEST(897) { CHECK(RunTest("l2v4", 897)); }
TEST(898) { CHECK(RunTest("l2v4", 898)); }
TEST(899) { CHECK(RunTest("l2v4", 899)); }
TEST(900) { CHECK(RunTest("l2v4", 900)); }
TEST(901) { CHECK(RunTest("l2v4", 901)); }
TEST(902) { CHECK(RunTest("l2v4", 902)); }
TEST(903) { CHECK(RunTest("l2v4", 903)); }
TEST(904) { CHECK(RunTest("l2v4", 904)); }
TEST(905) { CHECK(RunTest("l2v4", 905)); }
TEST(906) { CHECK(RunTest("l2v4", 906)); }
TEST(907) { CHECK(RunTest("l2v4", 907)); }
TEST(908) { CHECK(RunTest("l2v4", 908)); }
TEST(909) { CHECK(RunTest("l2v4", 909)); }
TEST(910) { CHECK(RunTest("l2v4", 910)); }
TEST(911) { CHECK(RunTest("l2v4", 911)); }
TEST(912) { CHECK(RunTest("l2v4", 912)); }
TEST(913) { CHECK(RunTest("l2v4", 913)); }
TEST(914) { CHECK(RunTest("l2v4", 914)); }
TEST(915) { CHECK(RunTest("l2v4", 915)); }
TEST(916) { CHECK(RunTest("l2v4", 916)); }
TEST(917) { CHECK(RunTest("l2v4", 917)); }
TEST(918) { CHECK(RunTest("l2v4", 918)); }
TEST(919) { CHECK(RunTest("l2v4", 919)); }
TEST(920) { CHECK(RunTest("l2v4", 920)); }
TEST(921) { CHECK(RunTest("l2v4", 921)); }
TEST(922) { CHECK(RunTest("l2v4", 922)); }
TEST(923) { CHECK(RunTest("l2v4", 923)); }
TEST(924) { CHECK(RunTest("l2v4", 924)); }
TEST(925) { CHECK(RunTest("l2v4", 925)); }
TEST(926) { CHECK(RunTest("l2v4", 926)); }
TEST(927) { CHECK(RunTest("l2v4", 927)); }
TEST(928) { CHECK(RunTest("l3v1", 928)); }
TEST(929) { CHECK(RunTest("l3v1", 929)); }
TEST(930) { CHECK(RunTest("l3v1", 930)); }
TEST(931) { CHECK(RunTest("l3v1", 931)); }
TEST(932) { CHECK(RunTest("l3v1", 932)); }
TEST(934) { CHECK(RunTest("l3v1", 934)); }
TEST(935) { CHECK(RunTest("l3v1", 935)); }
TEST(936) { CHECK(RunTest("l2v4", 936)); }
//TEST(937) { CHECK(RunTest("l2v4", 937)); } // CSymbolDelay
//TEST(938) { CHECK(RunTest("l2v4", 938)); } // CSymbolDelay
//TEST(939) { CHECK(RunTest("l2v4", 939)); } // CSymbolDelay
//TEST(940) { CHECK(RunTest("l2v4", 940)); } // CSymbolDelay
//TEST(941) { CHECK(RunTest("l2v4", 941)); } // CSymbolDelay
//TEST(942) { CHECK(RunTest("l2v4", 942)); } // CSymbolDelay
//TEST(943) { CHECK(RunTest("l2v4", 943)); } // CSymbolDelay
TEST(944) { CHECK(RunTest("l2v4", 944)); }
TEST(945) { CHECK(RunTest("l2v4", 945)); }
TEST(946) { CHECK(RunTest("l2v4", 946)); }
TEST(947) { CHECK(RunTest("l2v4", 947)); }
TEST(948) { CHECK(RunTest("l2v4", 948)); }
TEST(949) { CHECK(RunTest("l2v4", 949)); }
//TEST(950) { CHECK(RunTest("l2v4", 950)); }
//TEST(951) { CHECK(RunTest("l2v4", 951)); }
TEST(952) { CHECK(RunTest("l3v1", 952)); } 
TEST(953) { CHECK(RunTest("l3v1", 953)); }
TEST(954) { CHECK(RunTest("l2v4", 954)); }
TEST(955) { CHECK(RunTest("l2v4", 955)); }
TEST(956) { CHECK(RunTest("l2v4", 956)); }
TEST(957) { CHECK(RunTest("l2v4", 957)); }
TEST(958) { CHECK(RunTest("l2v4", 958)); }
TEST(959) { CHECK(RunTest("l2v4", 959)); }
TEST(960) { CHECK(RunTest("l3v1", 960)); }
TEST(961) { CHECK(RunTest("l3v1", 961)); }
TEST(962) { CHECK(RunTest("l3v1", 962)); }
TEST(963) { CHECK(RunTest("l3v1", 963)); }
TEST(964) { CHECK(RunTest("l3v1", 964)); }
TEST(965) { CHECK(RunTest("l3v1", 965)); } 
TEST(966) { CHECK(RunTest("l3v1", 966)); }
TEST(967) { CHECK(RunTest("l3v1", 967)); }
TEST(968) { CHECK(RunTest("l2v4", 968)); }
TEST(969) { CHECK(RunTest("l3v1", 969)); }
TEST(970) { CHECK(RunTest("l3v1", 970)); }
TEST(971) { CHECK(RunTest("l3v1", 971)); }
TEST(972) { CHECK(RunTest("l3v1", 972)); }
TEST(973) { CHECK(RunTest("l2v4", 973)); }
TEST(974) { CHECK(RunTest("l3v1", 974)); }
TEST(975) { CHECK(RunTest("l3v1", 975)); }
TEST(976) { CHECK(RunTest("l3v1", 976)); }
TEST(977) { CHECK(RunTest("l3v1", 977)); }
TEST(978) { CHECK(RunTest("l3v1", 978)); } //Crash in legacy C model.
TEST(979) { CHECK(RunTest("l2v4", 979)); }
TEST(980) { CHECK(RunTest("l3v1", 980)); }
//TEST(981)  { CHECK(RunTest("l3v1", 981  )); } // CSymbolDelay
//TEST(982)  { CHECK(RunTest("l3v1", 982  )); } // CSymbolDelay
//TEST(983)  { CHECK(RunTest("l3v1", 983  )); } // AlgebraicRule // CSymbolDelay
//TEST(984)  { CHECK(RunTest("l3v1", 984  )); } // CSymbolDelay
//TEST(985)  { CHECK(RunTest("l3v1", 985  )); } // CSymbolDelay
//TEST(986)  { CHECK(RunTest("l3v1", 986  )); } // FastReaction
//TEST(987)  { CHECK(RunTest("l3v1", 987  )); } // FastReaction
//TEST(988)  { CHECK(RunTest("l3v1", 988  )); } // FastReaction
TEST(989)  { CHECK(RunTest("l2v4", 989  )); }
TEST(990)  { CHECK(RunTest("l2v4", 990  )); }
TEST(991)  { CHECK(RunTest("l2v4", 991  )); }
TEST(992)  { CHECK(RunTest("l2v4", 992  )); }
//TEST(993)  { CHECK(RunTest("l2v4", 993  )); } // AlgebraicRule
TEST(994)  { CHECK(RunTest("l2v4", 994  )); }
TEST(995)  { CHECK(RunTest("l3v1", 995  )); }
TEST(996)  { CHECK(RunTest("l3v1", 996  )); }
TEST(997)  { CHECK(RunTest("l3v1", 997  )); }
TEST(998)  { CHECK(RunTest("l3v1", 998  )); }
TEST(999)  { CHECK(RunTest("l3v1", 999  )); }
TEST(1000) { CHECK(RunTest("l3v1", 1000 )); }
TEST(1001) { CHECK(RunTest("l3v1", 1001 )); }
TEST(1002) { CHECK(RunTest("l3v1", 1002 )); }
TEST(1003) { CHECK(RunTest("l3v1", 1003 )); }
TEST(1004) { CHECK(RunTest("l3v1", 1004 )); }
TEST(1005) { CHECK(RunTest("l3v1", 1005 )); }
TEST(1006) { CHECK(RunTest("l3v1", 1006 )); }
TEST(1007) { CHECK(RunTest("l3v1", 1007 )); }
TEST(1008) { CHECK(RunTest("l3v1", 1008 )); }
TEST(1009) { CHECK(RunTest("l3v1", 1009 )); }
TEST(1010) { CHECK(RunTest("l3v1", 1010 )); }
TEST(1011) { CHECK(RunTest("l3v1", 1011 )); }
TEST(1012) { CHECK(RunTest("l3v1", 1012 )); }
TEST(1013) { CHECK(RunTest("l3v1", 1013 )); }
TEST(1014) { CHECK(RunTest("l3v1", 1014 )); }
TEST(1015) { CHECK(RunTest("l3v1", 1015 )); }
TEST(1016) { CHECK(RunTest("l3v1", 1016 )); }
TEST(1017) { CHECK(RunTest("l3v1", 1017 )); }
TEST(1018) { CHECK(RunTest("l3v1", 1018 )); }
TEST(1019) { CHECK(RunTest("l3v1", 1019 )); }
TEST(1020) { CHECK(RunTest("l3v1", 1020 )); }
TEST(1021) { CHECK(RunTest("l3v1", 1021 )); }
TEST(1022) { CHECK(RunTest("l3v1", 1022 )); }
TEST(1023) { CHECK(RunTest("l3v1", 1023 )); }
TEST(1024) { CHECK(RunTest("l3v1", 1024 )); }
TEST(1025) { CHECK(RunTest("l3v1", 1025 )); }
TEST(1026) { CHECK(RunTest("l3v1", 1026 )); }
TEST(1027) { CHECK(RunTest("l2v4", 1027 )); }
TEST(1028) { CHECK(RunTest("l2v4", 1028 )); }
TEST(1029) { CHECK(RunTest("l2v4", 1029 )); }
TEST(1030) { CHECK(RunTest("l3v1", 1030 )); }
TEST(1031) { CHECK(RunTest("l3v1", 1031 )); }
TEST(1032) { CHECK(RunTest("l3v1", 1032 )); }
TEST(1033) { CHECK(RunTest("l3v1", 1033 )); }
TEST(1034) { CHECK(RunTest("l3v1", 1034 )); }
TEST(1035) { CHECK(RunTest("l3v1", 1035 )); }
TEST(1036) { CHECK(RunTest("l3v1", 1036 )); }
TEST(1037) { CHECK(RunTest("l3v1", 1037 )); }
TEST(1038) { CHECK(RunTest("l3v1", 1038 )); }
TEST(1039) { CHECK(RunTest("l3v1", 1039 )); }
TEST(1040) { CHECK(RunTest("l3v1", 1040 )); }
TEST(1041) { CHECK(RunTest("l3v1", 1041 )); }
TEST(1042) { CHECK(RunTest("l3v1", 1042 )); }
TEST(1043) { CHECK(RunTest("l3v1", 1043 )); }
//TEST(1044) { CHECK(RunTest("l3v1", 1044 )); } // AlgebraicRule
TEST(1045) { CHECK(RunTest("l3v1", 1045 )); }
TEST(1046) { CHECK(RunTest("l3v1", 1046 )); }
TEST(1047) { CHECK(RunTest("l3v1", 1047 )); }
TEST(1048) { CHECK(RunTest("l3v1", 1048 )); }
TEST(1049) { CHECK(RunTest("l3v1", 1049 )); }
TEST(1050) { CHECK(RunTest("l3v1", 1050 )); }
//TEST(1051) { CHECK(RunTest("l3v1", 1051 )); } // FastReaction
//TEST(1052) { CHECK(RunTest("l3v1", 1052 )); } // FastReaction
//TEST(1053) { CHECK(RunTest("l3v1", 1053 )); } // FastReaction
//TEST(1054) { CHECK(RunTest("l3v1", 1054 )); } // AlgebraicRule
TEST(1055) { CHECK(RunTest("l3v1", 1055 )); }
TEST(1056) { CHECK(RunTest("l3v1", 1056 )); }
TEST(1057) { CHECK(RunTest("l3v1", 1057 )); }
TEST(1058) { CHECK(RunTest("l3v1", 1058 )); }
TEST(1059) { CHECK(RunTest("l3v1", 1059 )); }
TEST(1060) { CHECK(RunTest("l3v1", 1060 )); }
TEST(1061) { CHECK(RunTest("l3v1", 1061 )); }
TEST(1062) { CHECK(RunTest("l3v1", 1062 )); }
TEST(1063) { CHECK(RunTest("l3v1", 1063 )); }
TEST(1064) { CHECK(RunTest("l3v1", 1064 )); }
TEST(1065) { CHECK(RunTest("l3v1", 1065 )); }
TEST(1066) { CHECK(RunTest("l3v1", 1066 )); }
TEST(1067) { CHECK(RunTest("l3v1", 1067 )); }
TEST(1068) { CHECK(RunTest("l3v1", 1068 )); }
TEST(1069) { CHECK(RunTest("l3v1", 1069 )); }
TEST(1070) { CHECK(RunTest("l3v1", 1070 )); }
TEST(1071) { CHECK(RunTest("l3v1", 1071 )); }
TEST(1072) { CHECK(RunTest("l3v1", 1072 )); }
TEST(1073) { CHECK(RunTest("l3v1", 1073 )); }
TEST(1074) { CHECK(RunTest("l3v1", 1074 )); }
TEST(1075) { CHECK(RunTest("l3v1", 1075 )); }
TEST(1076) { CHECK(RunTest("l3v1", 1076 )); }
TEST(1077) { CHECK(RunTest("l3v1", 1077 )); }
TEST(1078) { CHECK(RunTest("l3v1", 1078 )); }
TEST(1079) { CHECK(RunTest("l3v1", 1079 )); }
TEST(1080) { CHECK(RunTest("l3v1", 1080 )); }
TEST(1081) { CHECK(RunTest("l3v1", 1081 )); }
TEST(1082) { CHECK(RunTest("l3v1", 1082 )); }
//TEST(1083) { CHECK(RunTest("l3v1", 1083 )); } // AlgebraicRule
//TEST(1084) { CHECK(RunTest("l3v1", 1084 )); } // AlgebraicRule
//TEST(1085) { CHECK(RunTest("l3v1", 1085 )); } // AlgebraicRule
//TEST(1086) { CHECK(RunTest("l3v1", 1086 )); } // AlgebraicRule
TEST(1087) { CHECK(RunTest("l3v1", 1087 )); }
TEST(1088) { CHECK(RunTest("l3v1", 1088 )); }
TEST(1089) { CHECK(RunTest("l3v1", 1089 )); }
TEST(1090) { CHECK(RunTest("l3v1", 1090 )); }
TEST(1091) { CHECK(RunTest("l3v1", 1091 )); }
TEST(1092) { CHECK(RunTest("l3v1", 1092 )); }
TEST(1093) { CHECK(RunTest("l3v1", 1093 )); }
TEST(1094) { CHECK(RunTest("l3v1", 1094 )); }
TEST(1095) { CHECK(RunTest("l3v1", 1095 )); }
TEST(1096) { CHECK(RunTest("l3v1", 1096 )); }
TEST(1097) { CHECK(RunTest("l3v1", 1097 )); }
TEST(1098) { CHECK(RunTest("l3v1", 1098 )); }
TEST(1099) { CHECK(RunTest("l3v1", 1099 )); }
TEST(1100) { CHECK(RunTest("l3v1", 1100 )); }
TEST(1101) { CHECK(RunTest("l3v1", 1101 )); }
TEST(1102) { CHECK(RunTest("l3v1", 1102 )); }
TEST(1103) { CHECK(RunTest("l3v1", 1103 )); }
TEST(1104) { CHECK(RunTest("l3v1", 1104 )); }
TEST(1105) { CHECK(RunTest("l3v1", 1105 )); }
TEST(1106) { CHECK(RunTest("l3v1", 1106 )); }
TEST(1107) { CHECK(RunTest("l3v1", 1107 )); }
//TEST(1108) { CHECK(RunTest("l3v1", 1108 )); } // AlgebraicRule
TEST(1109) { CHECK(RunTest("l3v1", 1109 )); }
TEST(1110) { CHECK(RunTest("l3v1", 1110 )); }
TEST(1111) { CHECK(RunTest("l3v1", 1111 )); }
TEST(1112) { CHECK(RunTest("l3v1", 1112 )); }
TEST(1113) { CHECK(RunTest("l3v1", 1113 )); }
TEST(1114) { CHECK(RunTest("l3v1", 1114 )); }
TEST(1115) { CHECK(RunTest("l3v1", 1115 )); }
TEST(1116) { CHECK(RunTest("l3v1", 1116 )); }
TEST(1117) { CHECK(RunTest("l3v1", 1117 )); }
TEST(1118) { CHECK(RunTest("l3v1", 1118 )); }
TEST(1119) { CHECK(RunTest("l3v1", 1119 )); }
//TEST(1120) { CHECK(RunTest("l3v1", 1120 )); }
TEST(1121) { CHECK(RunTest("l3v1", 1121 )); }
TEST(1122) { CHECK(RunTest("l3v1", 1122 )); }
TEST(1123) { CHECK(RunTest("l3v1", 1123 )); }
/*
 * tests 1124 -> 1183 use the composite extension. 
 * work is underway to support this extension, but 
 * currently not supported.
 */
TEST(1184) { CHECK(RunTest("l3v1", 1184 )); }
TEST(1185) { CHECK(RunTest("l3v1", 1185 )); }
//TEST(1186) { CHECK(RunTest("l3v1", 1186 )); } // FBC
//TEST(1187) { CHECK(RunTest("l3v1", 1187 )); } // FBC
//TEST(1188) { CHECK(RunTest("l3v1", 1188 )); } // FBC
//TEST(1189) { CHECK(RunTest("l3v1", 1189 )); } // FBC
//TEST(1190) { CHECK(RunTest("l3v1", 1190 )); } // FBC
//TEST(1191) { CHECK(RunTest("l3v1", 1191 )); } // FBC
//TEST(1192) { CHECK(RunTest("l3v1", 1192 )); } // FBC
//TEST(1193) { CHECK(RunTest("l3v1", 1193 )); } // FBC
//TEST(1194) { CHECK(RunTest("l3v1", 1194 )); } // FBC
//TEST(1195) { CHECK(RunTest("l3v1", 1195 )); } // FBC
//TEST(1196) { CHECK(RunTest("l3v1", 1196 )); } // FBC

}


SUITE(SBML_TEST_SUITE_COMP)
{
TEST(1124) { CHECK(RunTest("l3v1", 1124 )); }
TEST(1125) { CHECK(RunTest("l3v1", 1125 )); }
TEST(1126) { CHECK(RunTest("l3v1", 1126 )); }
TEST(1127) { CHECK(RunTest("l3v1", 1127 )); }
//TEST(1128) { CHECK(RunTest("l3v1", 1128 )); } // comp
TEST(1129) { CHECK(RunTest("l3v1", 1129 )); }
TEST(1130) { CHECK(RunTest("l3v1", 1130 )); }
TEST(1131) { CHECK(RunTest("l3v1", 1131 )); }
TEST(1132) { CHECK(RunTest("l3v1", 1132 )); }
TEST(1133) { CHECK(RunTest("l3v1", 1133 )); }
TEST(1134) { CHECK(RunTest("l3v1", 1134 )); }
//TEST(1135) { CHECK(RunTest("l3v1", 1135 )); } // comp
//TEST(1136) { CHECK(RunTest("l3v1", 1136 )); } // comp
TEST(1137) { CHECK(RunTest("l3v1", 1137 )); }
TEST(1138) { CHECK(RunTest("l3v1", 1138 )); }
// TEST(1139) { CHECK(RunTest("l3v1", 1139 ));} // comp
// TEST(1140) { CHECK(RunTest("l3v1", 1140 ));} // comp
// TEST(1141) { CHECK(RunTest("l3v1", 1141 ));} // comp
//TEST(1142) { CHECK(RunTest("l3v1", 1142 )); } // AlgebraicRule // CSymbolDelay
TEST(1143) { CHECK(RunTest("l3v1", 1143 )); }
TEST(1144) { CHECK(RunTest("l3v1", 1144 )); }
TEST(1145) { CHECK(RunTest("l3v1", 1145 )); }
TEST(1146) { CHECK(RunTest("l3v1", 1146 )); }
TEST(1147) { CHECK(RunTest("l3v1", 1147 )); }
//TEST(1148) { CHECK(RunTest("l3v1", 1148 )); }
TEST(1149) { CHECK(RunTest("l3v1", 1149 )); }
TEST(1150) { CHECK(RunTest("l3v1", 1150 )); }
TEST(1151) { CHECK(RunTest("l3v1", 1151 )); }
// TEST(1152) { CHECK(RunTest("l3v1", 1152 ));} // comp
TEST(1153) { CHECK(RunTest("l3v1", 1153 )); }
TEST(1154) { CHECK(RunTest("l3v1", 1154 )); }
TEST(1155) { CHECK(RunTest("l3v1", 1155 )); }
TEST(1156) { CHECK(RunTest("l3v1", 1156 )); }
TEST(1157) { CHECK(RunTest("l3v1", 1157 )); }
TEST(1158) { CHECK(RunTest("l3v1", 1158 )); }
TEST(1159) { CHECK(RunTest("l3v1", 1159 )); }
TEST(1160) { CHECK(RunTest("l3v1", 1160 )); }
TEST(1161) { CHECK(RunTest("l3v1", 1161 )); }
TEST(1162) { CHECK(RunTest("l3v1", 1162 )); }
TEST(1163) { CHECK(RunTest("l3v1", 1163 )); }
TEST(1164) { CHECK(RunTest("l3v1", 1164 )); }
TEST(1165) { CHECK(RunTest("l3v1", 1165 )); }
TEST(1166) { CHECK(RunTest("l3v1", 1166 )); }
TEST(1167) { CHECK(RunTest("l3v1", 1167 )); }
TEST(1168) { CHECK(RunTest("l3v1", 1168 )); }
TEST(1169) { CHECK(RunTest("l3v1", 1169 )); }
//TEST(1170) { CHECK(RunTest("l3v1", 1170 )); }
TEST(1171) { CHECK(RunTest("l3v1", 1171 )); }
//TEST(1172) { CHECK(RunTest("l3v1", 1172 )); } // comp
//TEST(1173) { CHECK(RunTest("l3v1", 1173 )); } // CSymbolDelay
//TEST(1174) { CHECK(RunTest("l3v1", 1174 )); } // AlgebraicRule // CSymbolDelay
//TEST(1175) { CHECK(RunTest("l3v1", 1175 )); }
//TEST(1176) { CHECK(RunTest("l3v1", 1176 )); } // CSymbolDelay
//TEST(1177) { CHECK(RunTest("l3v1", 1177 )); } // comp
TEST(1178) { CHECK(RunTest("l3v1", 1178 )); }
//TEST(1179) { CHECK(RunTest("l3v1", 1179 )); } // comp
TEST(1180) { CHECK(RunTest("l3v1", 1180 )); }
TEST(1181) { CHECK(RunTest("l3v1", 1181 )); }
TEST(1182) { CHECK(RunTest("l3v1", 1182 )); }
TEST(1183) { CHECK(RunTest("l3v1", 1183 )); }
}


/**
 * The original test suite used for Valgrind testing
 */
SUITE(SBML_TEST_SUITE_VG1)
{
    TEST(15)  { CHECK(RunTest("l2v4", 15)); }
    TEST(16)  { CHECK(RunTest("l2v4", 16)); }
    TEST(18)  { CHECK(RunTest("l2v4", 18)); }
    TEST(19)  { CHECK(RunTest("l2v4", 19)); }
    TEST(27)  { CHECK(RunTest("l2v4", 27)); }
    TEST(29)  { CHECK(RunTest("l2v4", 29)); }
    TEST(31)  { CHECK(RunTest("l2v4", 31)); }
    TEST(889) { CHECK(RunTest("l2v4", 889)); }
    TEST(890) { CHECK(RunTest("l2v4", 890)); }
    TEST(891) { CHECK(RunTest("l2v4", 891)); }
    TEST(892) { CHECK(RunTest("l2v4", 892)); }
    TEST(893) { CHECK(RunTest("l2v4", 893)); }
    TEST(894) { CHECK(RunTest("l2v4", 894)); }
    TEST(895) { CHECK(RunTest("l2v4", 895)); }
    TEST(896) { CHECK(RunTest("l2v4", 896)); }
    TEST(952) { CHECK(RunTest("l3v1", 952)); }
    TEST(408) { CHECK(RunTest("l2v4", 408)); }
    TEST(1183) { CHECK(RunTest("l3v1", 1183 )); }
}

/**
 * another suite with more rate rules
 */
SUITE(SBML_TEST_SUITE_VG2)
{
    TEST(953) { CHECK(RunTest("l3v1", 953)); }
    //TEST(17)    { CHECK(RunTest("l2v4", 17));    }
    //TEST(32)    { CHECK(RunTest("l2v4", 32));    }
    //TEST(86)    { CHECK(RunTest("l2v4", 86));    }
    //TEST(165)   { CHECK(RunTest("l2v4", 165));   }
    //TEST(889)   { CHECK(RunTest("l2v4", 889));   }
    //TEST(1046)  { CHECK(RunTest("l3v1", 1046 )); }
}


SUITE(SBML_TEST_SUITE_RATE_RULES)
{
   TEST(31)  { CHECK(RunTest("l2v4", 31)); }
   TEST(32)  { CHECK(RunTest("l2v4", 32)); }
   TEST(33)  { CHECK(RunTest("l2v4", 33)); }
   //TEST(40)  { CHECK(RunTest("l2v4", 40)); }
   TEST(51)  { CHECK(RunTest("l2v4", 51)); }
   TEST(52)  { CHECK(RunTest("l2v4", 52)); }
   TEST(53)  { CHECK(RunTest("l2v4", 53)); }
   TEST(170)  { CHECK(RunTest("l2v4", 170)); }
   TEST(185)  { CHECK(RunTest("l2v4", 185)); }
   TEST(310)  { CHECK(RunTest("l2v4", 310)); }
   TEST(316)  { CHECK(RunTest("l2v4", 316)); }
   TEST(461)  { CHECK(RunTest("l2v4", 461)); }
   TEST(712)  { CHECK(RunTest("l2v4", 712)); }
}

SUITE(SBML_TEST_SUITE_EVENTS)
{
    TEST(26) { CHECK(RunTest("l2v4", 26)); }
    TEST(41) { CHECK(RunTest("l2v4", 41)); }
    TEST(71) { CHECK(RunTest("l2v4", 71)); }
    TEST(72) { CHECK(RunTest("l2v4", 72)); }
    TEST(73) { CHECK(RunTest("l2v4", 73)); }
    TEST(74) { CHECK(RunTest("l2v4", 74)); }

    TEST(351) { CHECK(RunTest("l2v4", 351)); }
    TEST(352) { CHECK(RunTest("l2v4", 352)); }
    TEST(353) { CHECK(RunTest("l2v4", 353)); }
    TEST(354) { CHECK(RunTest("l2v4", 354)); }
    TEST(355) { CHECK(RunTest("l2v4", 355)); }
    TEST(356) { CHECK(RunTest("l2v4", 356)); }
    TEST(357) { CHECK(RunTest("l2v4", 357)); }
    TEST(358) { CHECK(RunTest("l2v4", 358)); }
    TEST(359) { CHECK(RunTest("l2v4", 359)); }
    TEST(360) { CHECK(RunTest("l2v4", 360)); }
    TEST(361) { CHECK(RunTest("l2v4", 361)); }
    TEST(362) { CHECK(RunTest("l2v4", 362)); }
    TEST(363) { CHECK(RunTest("l2v4", 363)); }
    TEST(364) { CHECK(RunTest("l2v4", 364)); }
    TEST(365) { CHECK(RunTest("l2v4", 365)); }
    TEST(366) { CHECK(RunTest("l2v4", 366)); }
    TEST(367) { CHECK(RunTest("l2v4", 367)); }
    TEST(368) { CHECK(RunTest("l2v4", 368)); }
    TEST(369) { CHECK(RunTest("l2v4", 369)); }
    TEST(370) { CHECK(RunTest("l2v4", 370)); }
    TEST(371) { CHECK(RunTest("l2v4", 371)); }
    TEST(372) { CHECK(RunTest("l2v4", 372)); }
    TEST(373) { CHECK(RunTest("l2v4", 373)); }
    TEST(374) { CHECK(RunTest("l2v4", 374)); }
    TEST(375) { CHECK(RunTest("l2v4", 375)); }
    TEST(376) { CHECK(RunTest("l2v4", 376)); }
    TEST(377) { CHECK(RunTest("l2v4", 377)); }
    TEST(378) { CHECK(RunTest("l2v4", 378)); }
    TEST(379) { CHECK(RunTest("l2v4", 379)); }
    TEST(380) { CHECK(RunTest("l2v4", 380)); }
    TEST(381) { CHECK(RunTest("l2v4", 381)); }
    TEST(382) { CHECK(RunTest("l2v4", 382)); }
    TEST(383) { CHECK(RunTest("l2v4", 383)); }
    TEST(384) { CHECK(RunTest("l2v4", 384)); }
    TEST(385) { CHECK(RunTest("l2v4", 385)); }
    TEST(386) { CHECK(RunTest("l2v4", 386)); }
    TEST(387) { CHECK(RunTest("l2v4", 387)); }
    TEST(388) { CHECK(RunTest("l2v4", 388)); }
    TEST(389) { CHECK(RunTest("l2v4", 389)); }
    TEST(390) { CHECK(RunTest("l2v4", 390)); }
    TEST(391) { CHECK(RunTest("l2v4", 391)); }
    TEST(392) { CHECK(RunTest("l2v4", 392)); }
    TEST(393) { CHECK(RunTest("l2v4", 393)); }
    TEST(394) { CHECK(RunTest("l2v4", 394)); }
    TEST(395) { CHECK(RunTest("l2v4", 395)); }
    TEST(396) { CHECK(RunTest("l2v4", 396)); }
    TEST(397) { CHECK(RunTest("l2v4", 397)); }
    TEST(398) { CHECK(RunTest("l2v4", 398)); }
    TEST(399) { CHECK(RunTest("l2v4", 399)); }
    TEST(400) { CHECK(RunTest("l2v4", 400)); }
    TEST(401) { CHECK(RunTest("l2v4", 401)); }
    TEST(402) { CHECK(RunTest("l2v4", 402)); }
    TEST(403) { CHECK(RunTest("l2v4", 403)); }
    TEST(404) { CHECK(RunTest("l2v4", 404)); }
    TEST(405) { CHECK(RunTest("l2v4", 405)); }
    TEST(406) { CHECK(RunTest("l2v4", 406)); }
    TEST(407) { CHECK(RunTest("l2v4", 407)); }
    TEST(408) { CHECK(RunTest("l2v4", 408)); }
    TEST(409) { CHECK(RunTest("l2v4", 409)); }
    TEST(410) { CHECK(RunTest("l2v4", 410)); }
    TEST(411) { CHECK(RunTest("l2v4", 411)); }
    TEST(412) { CHECK(RunTest("l2v4", 412)); }
    TEST(413) { CHECK(RunTest("l2v4", 413)); }
    TEST(414) { CHECK(RunTest("l2v4", 414)); }
    TEST(415) { CHECK(RunTest("l2v4", 415)); }
    TEST(416) { CHECK(RunTest("l2v4", 416)); }
    TEST(417) { CHECK(RunTest("l2v4", 417)); }
    TEST(418) { CHECK(RunTest("l2v4", 418)); }
    TEST(419) { CHECK(RunTest("l2v4", 419)); }
    TEST(420) { CHECK(RunTest("l2v4", 420)); }
    TEST(421) { CHECK(RunTest("l2v4", 421)); }
    TEST(422) { CHECK(RunTest("l2v4", 422)); }
    TEST(423) { CHECK(RunTest("l2v4", 423)); }
    TEST(424) { CHECK(RunTest("l2v4", 424)); }
    TEST(425) { CHECK(RunTest("l2v4", 425)); }
    TEST(426) { CHECK(RunTest("l2v4", 426)); }
    TEST(427) { CHECK(RunTest("l2v4", 427)); }
    TEST(428) { CHECK(RunTest("l2v4", 428)); }
    TEST(429) { CHECK(RunTest("l2v4", 429)); }
    TEST(430) { CHECK(RunTest("l2v4", 430)); }
    TEST(431) { CHECK(RunTest("l2v4", 431)); }
    TEST(432) { CHECK(RunTest("l2v4", 432)); }
    TEST(433) { CHECK(RunTest("l2v4", 433)); }
    TEST(434) { CHECK(RunTest("l2v4", 434)); }
    TEST(435) { CHECK(RunTest("l2v4", 435)); }
    TEST(436) { CHECK(RunTest("l2v4", 436)); }
    TEST(437) { CHECK(RunTest("l2v4", 437)); }
    TEST(438) { CHECK(RunTest("l2v4", 438)); }
    TEST(439) { CHECK(RunTest("l2v4", 439)); }
    TEST(440) { CHECK(RunTest("l2v4", 440)); }
    TEST(441) { CHECK(RunTest("l2v4", 441)); }
    TEST(442) { CHECK(RunTest("l2v4", 442)); }
    TEST(443) { CHECK(RunTest("l2v4", 443)); }
    TEST(444) { CHECK(RunTest("l2v4", 444)); }
    TEST(445) { CHECK(RunTest("l2v4", 445)); }
    TEST(446) { CHECK(RunTest("l2v4", 446)); }
    TEST(447) { CHECK(RunTest("l2v4", 447)); }
    TEST(448) { CHECK(RunTest("l2v4", 448)); }
    TEST(449) { CHECK(RunTest("l2v4", 449)); }
    TEST(450) { CHECK(RunTest("l2v4", 450)); }
    TEST(451) { CHECK(RunTest("l2v4", 451)); }
    TEST(452) { CHECK(RunTest("l2v4", 452)); }
    TEST(453) { CHECK(RunTest("l2v4", 453)); }
    TEST(454) { CHECK(RunTest("l2v4", 454)); }
    TEST(455) { CHECK(RunTest("l2v4", 455)); }
    TEST(456) { CHECK(RunTest("l2v4", 456)); }
    TEST(457) { CHECK(RunTest("l2v4", 457)); }
    TEST(458) { CHECK(RunTest("l2v4", 458)); }
    TEST(459) { CHECK(RunTest("l2v4", 459)); }
    TEST(460) { CHECK(RunTest("l2v4", 460)); }
    TEST(461) { CHECK(RunTest("l2v4", 461)); }
    TEST(619) { CHECK(RunTest("l2v4", 619)); }
    TEST(620) { CHECK(RunTest("l2v4", 620)); }
    TEST(621) { CHECK(RunTest("l2v4", 621)); }
    TEST(622) { CHECK(RunTest("l2v4", 622)); }
    TEST(623) { CHECK(RunTest("l2v4", 623)); }
    TEST(624) { CHECK(RunTest("l2v4", 624)); }
    TEST(634) { CHECK(RunTest("l2v4", 634)); }
    TEST(635) { CHECK(RunTest("l2v4", 635)); }
    TEST(636) { CHECK(RunTest("l2v4", 636)); }
    TEST(637) { CHECK(RunTest("l2v4", 637)); }
    TEST(638) { CHECK(RunTest("l2v4", 638)); }
    TEST(639) { CHECK(RunTest("l2v4", 639)); }
    TEST(646) { CHECK(RunTest("l2v4", 646)); }
    TEST(647) { CHECK(RunTest("l2v4", 647)); }
    TEST(648) { CHECK(RunTest("l2v4", 648)); }
    TEST(649) { CHECK(RunTest("l2v4", 649)); }
    TEST(650) { CHECK(RunTest("l2v4", 650)); }
    TEST(651) { CHECK(RunTest("l2v4", 651)); }
    TEST(652) { CHECK(RunTest("l2v4", 652)); }
    TEST(653) { CHECK(RunTest("l2v4", 653)); }
    TEST(654) { CHECK(RunTest("l2v4", 654)); }
    TEST(655) { CHECK(RunTest("l2v4", 655)); }
    TEST(656) { CHECK(RunTest("l2v4", 656)); }
    TEST(657) { CHECK(RunTest("l2v4", 657)); }
    TEST(679) { CHECK(RunTest("l2v4", 679)); }
    TEST(680) { CHECK(RunTest("l2v4", 680)); }
    TEST(681) { CHECK(RunTest("l2v4", 681)); }
    TEST(682) { CHECK(RunTest("l2v4", 682)); }
    TEST(683) { CHECK(RunTest("l2v4", 683)); }
    TEST(684) { CHECK(RunTest("l2v4", 684)); }
    TEST(689) { CHECK(RunTest("l2v4", 689)); }
    TEST(690) { CHECK(RunTest("l2v4", 690)); }
    TEST(700) { CHECK(RunTest("l2v4", 700)); }
    TEST(701) { CHECK(RunTest("l2v4", 701)); }
    TEST(702) { CHECK(RunTest("l2v4", 702)); }
    TEST(707) { CHECK(RunTest("l2v4", 707)); }
    TEST(708) { CHECK(RunTest("l2v4", 708)); }
    TEST(723) { CHECK(RunTest("l2v4", 723)); }
    TEST(724) { CHECK(RunTest("l2v4", 724)); }
    TEST(730) { CHECK(RunTest("l2v4", 730)); }
    TEST(731) { CHECK(RunTest("l2v4", 731)); }
    TEST(736) { CHECK(RunTest("l2v4", 736)); }
    TEST(737) { CHECK(RunTest("l2v4", 737)); }
    TEST(743) { CHECK(RunTest("l2v4", 743)); }
    TEST(744) { CHECK(RunTest("l2v4", 744)); }
    TEST(745) { CHECK(RunTest("l2v4", 745)); }
    TEST(746) { CHECK(RunTest("l2v4", 746)); }
    TEST(747) { CHECK(RunTest("l2v4", 747)); }
    TEST(748) { CHECK(RunTest("l2v4", 748)); } 
    TEST(749) { CHECK(RunTest("l2v4", 749)); }
    TEST(750) { CHECK(RunTest("l2v4", 750)); }
    TEST(751) { CHECK(RunTest("l2v4", 751)); }
    TEST(752) { CHECK(RunTest("l2v4", 752)); }
    TEST(753) { CHECK(RunTest("l2v4", 753)); }
    TEST(754) { CHECK(RunTest("l2v4", 754)); }
    TEST(755) { CHECK(RunTest("l2v4", 755)); }
    TEST(756) { CHECK(RunTest("l2v4", 756)); }
    TEST(757) { CHECK(RunTest("l2v4", 757)); }
    TEST(758) { CHECK(RunTest("l2v4", 758)); }
    TEST(759) { CHECK(RunTest("l2v4", 759)); }
 //TEST(760) { CHECK(RunTest("l2v4", 760)); } // AlgebraicRule
 //TEST(761) { CHECK(RunTest("l2v4", 761)); } // AlgebraicRule
 //TEST(762) { CHECK(RunTest("l2v4", 762)); } // AlgebraicRule
    TEST(763) { CHECK(RunTest("l2v4", 763)); }
    TEST(764) { CHECK(RunTest("l2v4", 764)); }
    TEST(765) { CHECK(RunTest("l2v4", 765)); }
    TEST(766) { CHECK(RunTest("l2v4", 766)); }
    TEST(767) { CHECK(RunTest("l2v4", 767)); }
    TEST(768) { CHECK(RunTest("l2v4", 768)); }
    TEST(769) { CHECK(RunTest("l2v4", 769)); }
    TEST(770) { CHECK(RunTest("l2v4", 770)); }
    TEST(771) { CHECK(RunTest("l2v4", 771)); }
    TEST(772) { CHECK(RunTest("l2v4", 772)); }
    TEST(773) { CHECK(RunTest("l2v4", 773)); }
    TEST(774) { CHECK(RunTest("l2v4", 774)); }
    TEST(775) { CHECK(RunTest("l2v4", 775)); }
    TEST(776) { CHECK(RunTest("l2v4", 776)); }


    TEST(952) { CHECK(RunTest("l3v1", 952)); } 
    TEST(953) { CHECK(RunTest("l3v1", 953)); }
    TEST(954) { CHECK(RunTest("l2v4", 954)); }
    TEST(955) { CHECK(RunTest("l2v4", 955)); }
    TEST(956) { CHECK(RunTest("l2v4", 956)); }
    TEST(957) { CHECK(RunTest("l2v4", 957)); }
    TEST(958) { CHECK(RunTest("l2v4", 958)); }
    TEST(959) { CHECK(RunTest("l2v4", 959)); }
    TEST(960) { CHECK(RunTest("l3v1", 960)); }
    TEST(961) { CHECK(RunTest("l3v1", 961)); }
    TEST(962) { CHECK(RunTest("l3v1", 962)); }


    TEST(963) { CHECK(RunTest("l3v1", 963)); }
    TEST(964) { CHECK(RunTest("l3v1", 964)); }
    TEST(965) { CHECK(RunTest("l3v1", 965)); } 
    TEST(966) { CHECK(RunTest("l3v1", 966)); }
    TEST(967) { CHECK(RunTest("l3v1", 967)); }
    TEST(968) { CHECK(RunTest("l2v4", 968)); }
}

SUITE(SBML_TEST_SUITE_LONGTIME)
{
    TEST(964) { CHECK(RunTest("l3v1", 964)); }
    TEST(952) { CHECK(RunTest("l3v1", 952)); }
    TEST(953) { CHECK(RunTest("l3v1", 953)); }
    TEST(954) { CHECK(RunTest("l2v4", 954)); }
    TEST(963) { CHECK(RunTest("l3v1", 963)); }
    TEST(221) { CHECK(RunTest("l2v4", 221)); }
    TEST(831) { CHECK(RunTest("l2v4", 831)); }
    TEST(326) { CHECK(RunTest("l2v4", 326)); }
    TEST(430) { CHECK(RunTest("l2v4", 430)); }
    TEST(110) { CHECK(RunTest("l2v4", 110)); }
    TEST(1116){ CHECK(RunTest("l3v1", 1116 )); }
    TEST(706) { CHECK(RunTest("l2v4", 706)); }
    TEST(579) { CHECK(RunTest("l2v4", 579)); }
    TEST(967) { CHECK(RunTest("l3v1", 967)); }
    TEST(965) { CHECK(RunTest("l3v1", 965)); }
    TEST(966) { CHECK(RunTest("l3v1", 966)); }
}


SUITE(SBML_TEST_SUITE_C_FAIL)
{
    TEST(39)  { CHECK_EQUAL(false, RunTest("l2v4", 39));  } // AlgebraicRule
    TEST(40)  { CHECK_EQUAL(false, RunTest("l2v4", 40));  } // AlgebraicRule
    TEST(182) { CHECK_EQUAL(false, RunTest("l2v4", 182)); } // AlgebraicRule
    TEST(184) { CHECK_EQUAL(false, RunTest("l2v4", 184)); } // AlgebraicRule
    TEST(531) { CHECK_EQUAL(false, RunTest("l2v4", 531)); } // AlgebraicRule
    TEST(533) { CHECK_EQUAL(false, RunTest("l2v4", 533)); } // AlgebraicRule
    TEST(534) { CHECK_EQUAL(false, RunTest("l2v4", 534)); } // AlgebraicRule
    TEST(535) { CHECK_EQUAL(false, RunTest("l2v4", 535)); } // AlgebraicRule
    TEST(536) { CHECK_EQUAL(false, RunTest("l2v4", 536)); } // AlgebraicRule
    TEST(537) { CHECK_EQUAL(false, RunTest("l2v4", 537)); } // AlgebraicRule
    TEST(538) { CHECK_EQUAL(false, RunTest("l2v4", 538)); } // AlgebraicRule
    TEST(543) { CHECK_EQUAL(false, RunTest("l2v4", 543)); } // AlgebraicRule
    TEST(546) { CHECK_EQUAL(false, RunTest("l2v4", 546)); } // AlgebraicRule
    TEST(548) { CHECK_EQUAL(false, RunTest("l2v4", 548)); } // AlgebraicRule
    TEST(549) { CHECK_EQUAL(false, RunTest("l2v4", 549)); } // AlgebraicRule
    TEST(550) { CHECK_EQUAL(false, RunTest("l2v4", 550)); } // AlgebraicRule
    TEST(551) { CHECK_EQUAL(false, RunTest("l2v4", 551)); } // AlgebraicRule
    TEST(552) { CHECK_EQUAL(false, RunTest("l2v4", 552)); } // AlgebraicRule
    TEST(553) { CHECK_EQUAL(false, RunTest("l2v4", 553)); } // AlgebraicRule
    TEST(554) { CHECK_EQUAL(false, RunTest("l2v4", 554)); } // AlgebraicRule
    TEST(555) { CHECK_EQUAL(false, RunTest("l2v4", 555)); } // AlgebraicRule
    TEST(556) { CHECK_EQUAL(false, RunTest("l2v4", 556)); } // AlgebraicRule
    TEST(557) { CHECK_EQUAL(false, RunTest("l2v4", 557)); } // AlgebraicRule
    TEST(558) { CHECK_EQUAL(false, RunTest("l2v4", 558)); } // AlgebraicRule
    TEST(559) { CHECK_EQUAL(false, RunTest("l2v4", 559)); } // AlgebraicRule
    TEST(560) { CHECK_EQUAL(false, RunTest("l2v4", 560)); } // AlgebraicRule
    TEST(561) { CHECK_EQUAL(false, RunTest("l2v4", 561)); } // AlgebraicRule
    TEST(562) { CHECK_EQUAL(false, RunTest("l2v4", 562)); } // AlgebraicRule
    TEST(563) { CHECK_EQUAL(false, RunTest("l2v4", 563)); } // AlgebraicRule
    TEST(564) { CHECK_EQUAL(false, RunTest("l2v4", 564)); } // AlgebraicRule
    TEST(565) { CHECK_EQUAL(false, RunTest("l2v4", 565)); } // AlgebraicRule
    TEST(566) { CHECK_EQUAL(false, RunTest("l2v4", 566)); } // AlgebraicRule
    TEST(567) { CHECK_EQUAL(false, RunTest("l2v4", 567)); } // AlgebraicRule
    TEST(569) { CHECK_EQUAL(false, RunTest("l2v4", 569)); } // AlgebraicRule
    TEST(570) { CHECK_EQUAL(false, RunTest("l2v4", 570)); } // AlgebraicRule
    TEST(571) { CHECK_EQUAL(false, RunTest("l2v4", 571)); } // AlgebraicRule
    TEST(573) { CHECK_EQUAL(false, RunTest("l2v4", 573)); } // AlgebraicRule
    TEST(575) { CHECK_EQUAL(false, RunTest("l2v4", 575)); } // AlgebraicRule
    TEST(576) { CHECK_EQUAL(false, RunTest("l2v4", 576)); } // AlgebraicRule

    TEST(613) { CHECK_EQUAL(false, RunTest("l2v4", 613)); } // AlgebraicRule
    TEST(614) { CHECK_EQUAL(false, RunTest("l2v4", 614)); } // AlgebraicRule
    TEST(615) { CHECK_EQUAL(false, RunTest("l2v4", 615)); } // AlgebraicRule

    TEST(628) { CHECK_EQUAL(false, RunTest("l2v4", 628)); } // AlgebraicRule
    TEST(629) { CHECK_EQUAL(false, RunTest("l2v4", 629)); } // AlgebraicRule
    TEST(630) { CHECK_EQUAL(false, RunTest("l2v4", 630)); } // AlgebraicRule

    TEST(658) { CHECK_EQUAL(false, RunTest("l2v4", 658)); } // AlgebraicRule
    TEST(659) { CHECK_EQUAL(false, RunTest("l2v4", 659)); } // AlgebraicRule
    TEST(660) { CHECK_EQUAL(false, RunTest("l2v4", 660)); } // AlgebraicRule
    TEST(661) { CHECK_EQUAL(false, RunTest("l2v4", 661)); } // AlgebraicRule
    TEST(662) { CHECK_EQUAL(false, RunTest("l2v4", 662)); } // AlgebraicRule
    TEST(663) { CHECK_EQUAL(false, RunTest("l2v4", 663)); } // AlgebraicRule
    TEST(664) { CHECK_EQUAL(false, RunTest("l2v4", 664)); } // AlgebraicRule
    TEST(665) { CHECK_EQUAL(false, RunTest("l2v4", 665)); } // AlgebraicRule
    TEST(666) { CHECK_EQUAL(false, RunTest("l2v4", 666)); } // AlgebraicRule

    TEST(673) { CHECK_EQUAL(false, RunTest("l2v4", 673)); } // AlgebraicRule
    TEST(674) { CHECK_EQUAL(false, RunTest("l2v4", 674)); } // AlgebraicRule
    TEST(675) { CHECK_EQUAL(false, RunTest("l2v4", 675)); } // AlgebraicRule

    TEST(687) { CHECK_EQUAL(false, RunTest("l2v4", 687)); } // AlgebraicRule

    TEST(695) { CHECK_EQUAL(false, RunTest("l2v4", 695)); } // AlgebraicRule
    TEST(696) { CHECK_EQUAL(false, RunTest("l2v4", 696)); } // AlgebraicRule

    TEST(705) { CHECK_EQUAL(false, RunTest("l2v4", 705)); } // AlgebraicRule

    TEST(760) { CHECK_EQUAL(false, RunTest("l2v4", 760)); } // AlgebraicRule
    TEST(761) { CHECK_EQUAL(false, RunTest("l2v4", 761)); } // AlgebraicRule
    TEST(762) { CHECK_EQUAL(false, RunTest("l2v4", 762)); } // AlgebraicRule

    TEST(777) { CHECK_EQUAL(false, RunTest("l2v4", 777)); } // AlgebraicRule
    TEST(778) { CHECK_EQUAL(false, RunTest("l2v4", 778)); } // AlgebraicRule
    TEST(779) { CHECK_EQUAL(false, RunTest("l2v4", 779)); } // AlgebraicRule
    TEST(780) { CHECK_EQUAL(false, RunTest("l2v4", 780)); } // AlgebraicRule

    TEST(844) { CHECK_EQUAL(false, RunTest("l2v4", 844)); } // AlgebraicRule

    TEST(870) { CHECK_EQUAL(false, RunTest("l2v4", 870)); } // FastReaction
    TEST(871) { CHECK_EQUAL(false, RunTest("l2v4", 871)); } // FastReaction
    TEST(872) { CHECK_EQUAL(false, RunTest("l2v4", 872)); } // FastReaction
    TEST(873) { CHECK_EQUAL(false, RunTest("l2v4", 873)); } // FastReaction
    TEST(874) { CHECK_EQUAL(false, RunTest("l2v4", 874)); } // FastReaction
    TEST(875) { CHECK_EQUAL(false, RunTest("l2v4", 875)); } // FastReaction
    TEST(876) { CHECK_EQUAL(false, RunTest("l2v4", 876)); } // AlgebraicRule

    TEST(937) { CHECK_EQUAL(false, RunTest("l2v4", 937)); } // CSymbolDelay
    TEST(938) { CHECK_EQUAL(false, RunTest("l2v4", 938)); } // CSymbolDelay
    TEST(939) { CHECK_EQUAL(false, RunTest("l2v4", 939)); } // CSymbolDelay
    TEST(940) { CHECK_EQUAL(false, RunTest("l2v4", 940)); } // CSymbolDelay
    TEST(941) { CHECK_EQUAL(false, RunTest("l2v4", 941)); } // CSymbolDelay
    TEST(942) { CHECK_EQUAL(false, RunTest("l2v4", 942)); } // CSymbolDelay
    TEST(943) { CHECK_EQUAL(false, RunTest("l2v4", 943)); } // CSymbolDelay

    TEST(950) { CHECK_EQUAL(false, RunTest("l2v4", 950)); } // TODO:  HANDLE INF AND NAN IN INPUT AND OUTPUT
    TEST(951) { CHECK_EQUAL(false, RunTest("l2v4", 951)); } // TODO:  HANDLE INF AND NAN IN INPUT AND OUTPUT

    TEST(981)  { CHECK_EQUAL(false, RunTest("l3v1", 981  )); } // CSymbolDelay
    TEST(982)  { CHECK_EQUAL(false, RunTest("l3v1", 982  )); } // CSymbolDelay
    TEST(983)  { CHECK_EQUAL(false, RunTest("l3v1", 983  )); } // AlgebraicRule // CSymbolDelay
    TEST(984)  { CHECK_EQUAL(false, RunTest("l3v1", 984  )); } // CSymbolDelay
    TEST(985)  { CHECK_EQUAL(false, RunTest("l3v1", 985  )); } // CSymbolDelay
    TEST(986)  { CHECK_EQUAL(false, RunTest("l3v1", 986  )); } // FastReaction
    TEST(987)  { CHECK_EQUAL(false, RunTest("l3v1", 987  )); } // FastReaction
    TEST(988)  { CHECK_EQUAL(false, RunTest("l3v1", 988  )); } // FastReaction

    TEST(993)  { CHECK_EQUAL(false, RunTest("l2v4", 993  )); } // AlgebraicRule

    TEST(1044) { CHECK_EQUAL(false, RunTest("l3v1", 1044 )); } // AlgebraicRule

    TEST(1051) { CHECK_EQUAL(false, RunTest("l3v1", 1051 )); } // FastReaction
    TEST(1052) { CHECK_EQUAL(false, RunTest("l3v1", 1052 )); } // FastReaction
    TEST(1053) { CHECK_EQUAL(false, RunTest("l3v1", 1053 )); } // FastReaction
    TEST(1054) { CHECK_EQUAL(false, RunTest("l3v1", 1054 )); } // AlgebraicRule

    TEST(1083) { CHECK_EQUAL(false, RunTest("l3v1", 1083 )); } // AlgebraicRule
    TEST(1084) { CHECK_EQUAL(false, RunTest("l3v1", 1084 )); } // AlgebraicRule
    TEST(1085) { CHECK_EQUAL(false, RunTest("l3v1", 1085 )); } // AlgebraicRule
    TEST(1086) { CHECK_EQUAL(false, RunTest("l3v1", 1086 )); } // AlgebraicRule

    TEST(1108) { CHECK_EQUAL(false, RunTest("l3v1", 1108 )); } // AlgebraicRule

    TEST(1120) { CHECK_EQUAL(false, RunTest("l3v1", 1120 )); } // TODO:  ORDERING BUG
    TEST(1142) { CHECK_EQUAL(false, RunTest("l3v1", 1142 )); } // AlgebraicRule // CSymbolDelay

    TEST(1148) { CHECK_EQUAL(false, RunTest("l3v1", 1148 )); } // TODO:  NUMBERS SLIGHTLY OFF:  WHY?

    TEST(1170) { CHECK_EQUAL(false, RunTest("l3v1", 1170 )); } // TODO:  ORDERING BUG
    TEST(1173) { CHECK_EQUAL(false, RunTest("l3v1", 1173 )); } // CSymbolDelay
    TEST(1174) { CHECK_EQUAL(false, RunTest("l3v1", 1174 )); } // AlgebraicRule // CSymbolDelay
    TEST(1175) { CHECK_EQUAL(false, RunTest("l3v1", 1175 )); } // TODO:  ORDERING BUG
    TEST(1176) { CHECK_EQUAL(false, RunTest("l3v1", 1176 )); } // CSymbolDelay

    TEST(1186) { CHECK_EQUAL(false, RunTest("l3v1", 1186 )); } // FBC
    TEST(1187) { CHECK_EQUAL(false, RunTest("l3v1", 1187 )); } // FBC
    TEST(1188) { CHECK_EQUAL(false, RunTest("l3v1", 1188 )); } // FBC
    TEST(1189) { CHECK_EQUAL(false, RunTest("l3v1", 1189 )); } // FBC
    TEST(1190) { CHECK_EQUAL(false, RunTest("l3v1", 1190 )); } // FBC
    TEST(1191) { CHECK_EQUAL(false, RunTest("l3v1", 1191 )); } // FBC
    TEST(1192) { CHECK_EQUAL(false, RunTest("l3v1", 1192 )); } // FBC
    TEST(1193) { CHECK_EQUAL(false, RunTest("l3v1", 1193 )); } // FBC
    TEST(1194) { CHECK_EQUAL(false, RunTest("l3v1", 1194 )); } // FBC
    TEST(1195) { CHECK_EQUAL(false, RunTest("l3v1", 1195 )); } // FBC
    TEST(1196) { CHECK_EQUAL(false, RunTest("l3v1", 1196 )); } // FBC

}











