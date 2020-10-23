#include "gtest/gtest.h"
//#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrUtils.h"
#include "rrTestSuiteModelSimulation.h"
using namespace testing;
using namespace rr;
using namespace std;

extern string gRRTestDir;

bool RunTest(int number); //Runs both the first and the last version of the test.
bool RunTest(const string& version, int number);
bool CheckLoad(const string& version, int number);
void LoadAndSimulate(const string& version, int caseNumber, RoadRunner& rr, TestSuiteModelSimulation& simulation);

TEST(OneTest, 1)
{
    //ASSERT_TRUE(RunTest(1630)); //Use when need to run one test.
}

TEST(SBMLtestsuite, A)
{
    //ASSERT_TRUE(RunTest(1630)); //Use when need to ASSERT_TRUE one thing.
    ASSERT_TRUE(RunTest(1));
    ASSERT_TRUE(RunTest(2));
    ASSERT_TRUE(RunTest(3));
    ASSERT_TRUE(RunTest(4));
    ASSERT_TRUE(RunTest(5));
    ASSERT_TRUE(RunTest(6));
    ASSERT_TRUE(RunTest(7));
    ASSERT_TRUE(RunTest(8));
    ASSERT_TRUE(RunTest(9));
    ASSERT_TRUE(RunTest(10));
    ASSERT_TRUE(RunTest(11));
    ASSERT_TRUE(RunTest(12));
    ASSERT_TRUE(RunTest(13));
    ASSERT_TRUE(RunTest(14));
    ASSERT_TRUE(RunTest(15));
    ASSERT_TRUE(RunTest(16));
    ASSERT_TRUE(RunTest(17));
    ASSERT_TRUE(RunTest(18));
    ASSERT_TRUE(RunTest(19));
    ASSERT_TRUE(RunTest(20));
    ASSERT_TRUE(RunTest(21));
    ASSERT_TRUE(RunTest(22));
    ASSERT_TRUE(RunTest(23));
    ASSERT_TRUE(RunTest(24));
    ASSERT_TRUE(RunTest(25));
    ASSERT_TRUE(RunTest(26));
    ASSERT_TRUE(RunTest(27));
    ASSERT_TRUE(RunTest(28));
    ASSERT_TRUE(RunTest(29));
    ASSERT_TRUE(RunTest(30));
    ASSERT_TRUE(RunTest(31));
    ASSERT_TRUE(RunTest(32));
    ASSERT_TRUE(RunTest(33));
    ASSERT_TRUE(RunTest(34));
    ASSERT_TRUE(RunTest(35));
    ASSERT_TRUE(RunTest(36));
    ASSERT_TRUE(RunTest(37));
    ASSERT_TRUE(RunTest(38));
    ASSERT_TRUE(RunTest(39));
    ASSERT_TRUE(RunTest(40));
    ASSERT_TRUE(RunTest(41));
    ASSERT_TRUE(RunTest(42));
    ASSERT_TRUE(RunTest(43));
    ASSERT_TRUE(RunTest(44));
    ASSERT_TRUE(RunTest(45));
    ASSERT_TRUE(RunTest(46));
    ASSERT_TRUE(RunTest(47));
    ASSERT_TRUE(RunTest(48));
    ASSERT_TRUE(RunTest(49));
    ASSERT_TRUE(RunTest(50));
    ASSERT_TRUE(RunTest(51));
    ASSERT_TRUE(RunTest(52));
    ASSERT_TRUE(RunTest(53));
    ASSERT_TRUE(RunTest(54));
    ASSERT_TRUE(RunTest(55));
    ASSERT_TRUE(RunTest(56));
    ASSERT_TRUE(RunTest(57));
    ASSERT_TRUE(RunTest(58));
    ASSERT_TRUE(RunTest(59));
    ASSERT_TRUE(RunTest(60));
    ASSERT_TRUE(RunTest(61));
    ASSERT_TRUE(RunTest(62));
    ASSERT_TRUE(RunTest(63));
    ASSERT_TRUE(RunTest(64));
    ASSERT_TRUE(RunTest(65));
    ASSERT_TRUE(RunTest(66));
    ASSERT_TRUE(RunTest(67));
    ASSERT_TRUE(RunTest(68));
    ASSERT_TRUE(RunTest(69));
    ASSERT_TRUE(RunTest(70));
    ASSERT_TRUE(RunTest(71));
    ASSERT_TRUE(RunTest(72));
    ASSERT_TRUE(RunTest(73));
    ASSERT_TRUE(RunTest(74));
    ASSERT_TRUE(RunTest(75));
    ASSERT_TRUE(RunTest(76));
    ASSERT_TRUE(RunTest(77));
    ASSERT_TRUE(RunTest(78));
    ASSERT_TRUE(RunTest(79));
    ASSERT_TRUE(RunTest(80));
    ASSERT_TRUE(RunTest(81));
    ASSERT_TRUE(RunTest(82));
    ASSERT_TRUE(RunTest(83));
    ASSERT_TRUE(RunTest(84));
    ASSERT_TRUE(RunTest(85));
    ASSERT_TRUE(RunTest(86));
    ASSERT_TRUE(RunTest(87));
    ASSERT_TRUE(RunTest(88));
    ASSERT_TRUE(RunTest(89));
    ASSERT_TRUE(RunTest(90));
    ASSERT_TRUE(RunTest(91));
    ASSERT_TRUE(RunTest(92));
    ASSERT_TRUE(RunTest(93));
    ASSERT_TRUE(RunTest(94));
    ASSERT_TRUE(RunTest(95));
    ASSERT_TRUE(RunTest(96));
    ASSERT_TRUE(RunTest(97));
    ASSERT_TRUE(RunTest(98));
    ASSERT_TRUE(RunTest(99));
    ASSERT_TRUE(RunTest(100));
    ASSERT_TRUE(RunTest(101));
    ASSERT_TRUE(RunTest(102));
    ASSERT_TRUE(RunTest(103));
    ASSERT_TRUE(RunTest(104));
    ASSERT_TRUE(RunTest(105));
    ASSERT_TRUE(RunTest(106));
    ASSERT_TRUE(RunTest(107));
    ASSERT_TRUE(RunTest(108));
    ASSERT_TRUE(RunTest(109));
    ASSERT_TRUE(RunTest(110));
    ASSERT_TRUE(RunTest(111));
    ASSERT_TRUE(RunTest(112));
    ASSERT_TRUE(RunTest(113));
    ASSERT_TRUE(RunTest(114));
    ASSERT_TRUE(RunTest(115));
    ASSERT_TRUE(RunTest(116));
    ASSERT_TRUE(RunTest(117));
    ASSERT_TRUE(RunTest(118));
    ASSERT_TRUE(RunTest(119));
    ASSERT_TRUE(RunTest(120));
    ASSERT_TRUE(RunTest(121));
    ASSERT_TRUE(RunTest(122));
    ASSERT_TRUE(RunTest(123));
    ASSERT_TRUE(RunTest(124));
    ASSERT_TRUE(RunTest(125));
    ASSERT_TRUE(RunTest(126));
    ASSERT_TRUE(RunTest(127));
    ASSERT_TRUE(RunTest(128));
    ASSERT_TRUE(RunTest(129));
    ASSERT_TRUE(RunTest(130));
    ASSERT_TRUE(RunTest(131));
    ASSERT_TRUE(RunTest(132));
    ASSERT_TRUE(RunTest(133));
    ASSERT_TRUE(RunTest(134));
    ASSERT_TRUE(RunTest(135));
    ASSERT_TRUE(RunTest(136));
    ASSERT_TRUE(RunTest(137));
    ASSERT_TRUE(RunTest(138));
    ASSERT_TRUE(RunTest(139));
    ASSERT_TRUE(RunTest(140));
    ASSERT_TRUE(RunTest(141));
    ASSERT_TRUE(RunTest(142));
    ASSERT_TRUE(RunTest(143));
    ASSERT_TRUE(RunTest(144));
    ASSERT_TRUE(RunTest(145));
    ASSERT_TRUE(RunTest(146));
    ASSERT_TRUE(RunTest(147));
    ASSERT_TRUE(RunTest(148));
    ASSERT_TRUE(RunTest(149));
    ASSERT_TRUE(RunTest(150));
    ASSERT_TRUE(RunTest(151));
    ASSERT_TRUE(RunTest(152));
    ASSERT_TRUE(RunTest(153));
    ASSERT_TRUE(RunTest(154));
    ASSERT_TRUE(RunTest(155));
    ASSERT_TRUE(RunTest(156));
    ASSERT_TRUE(RunTest(157));
    ASSERT_TRUE(RunTest(158));
    ASSERT_TRUE(RunTest(159));
    ASSERT_TRUE(RunTest(160));
    ASSERT_TRUE(RunTest(161));
    ASSERT_TRUE(RunTest(162));
    ASSERT_TRUE(RunTest(163));
    ASSERT_TRUE(RunTest(164));
    ASSERT_TRUE(RunTest(165));
    ASSERT_TRUE(RunTest(166));
    ASSERT_TRUE(RunTest(167));
    ASSERT_TRUE(RunTest(168));
    ASSERT_TRUE(RunTest(169));
    ASSERT_TRUE(RunTest(170));
    ASSERT_TRUE(RunTest(171));
    ASSERT_TRUE(RunTest(172));
    ASSERT_TRUE(RunTest(173));
    ASSERT_TRUE(RunTest(174));
    ASSERT_TRUE(RunTest(175));
    ASSERT_TRUE(RunTest(176));
    ASSERT_TRUE(RunTest(177));
    ASSERT_TRUE(RunTest(178));
    ASSERT_TRUE(RunTest(179));
    ASSERT_TRUE(RunTest(180));
    ASSERT_TRUE(RunTest(181));
    ASSERT_TRUE(RunTest(182));
    ASSERT_TRUE(RunTest(183));
    ASSERT_TRUE(RunTest(184));
    ASSERT_TRUE(RunTest(185));
    ASSERT_TRUE(RunTest(186));
    ASSERT_TRUE(RunTest(187));
    ASSERT_TRUE(RunTest(188));
    ASSERT_TRUE(RunTest(189));
    ASSERT_TRUE(RunTest(190));
    ASSERT_TRUE(RunTest(191));
    ASSERT_TRUE(RunTest(192));
    ASSERT_TRUE(RunTest(193));
    ASSERT_TRUE(RunTest(194));
    ASSERT_TRUE(RunTest(195));
    ASSERT_TRUE(RunTest(196));
    ASSERT_TRUE(RunTest(197));
    ASSERT_TRUE(RunTest(198));
    ASSERT_TRUE(RunTest(199));
    ASSERT_TRUE(RunTest(200));
    ASSERT_TRUE(RunTest(201));
    ASSERT_TRUE(RunTest(202));
    ASSERT_TRUE(RunTest(203));
    ASSERT_TRUE(RunTest(204));
    ASSERT_TRUE(RunTest(205));
    ASSERT_TRUE(RunTest(206));
    ASSERT_TRUE(RunTest(207));
    ASSERT_TRUE(RunTest(208));
    ASSERT_TRUE(RunTest(209));
    ASSERT_TRUE(RunTest(210));
    ASSERT_TRUE(RunTest(211));
    ASSERT_TRUE(RunTest(212));
    ASSERT_TRUE(RunTest(213));
    ASSERT_TRUE(RunTest(214));
    ASSERT_TRUE(RunTest(215));
    ASSERT_TRUE(RunTest(216));
    ASSERT_TRUE(RunTest(217));
    ASSERT_TRUE(RunTest(218));
    ASSERT_TRUE(RunTest(219));
    ASSERT_TRUE(RunTest(220));
    ASSERT_TRUE(RunTest(221));
    ASSERT_TRUE(RunTest(222));
    ASSERT_TRUE(RunTest(223));
    ASSERT_TRUE(RunTest(224));
    ASSERT_TRUE(RunTest(225));
    ASSERT_TRUE(RunTest(226));
    ASSERT_TRUE(RunTest(227));
    ASSERT_TRUE(RunTest(228));
    ASSERT_TRUE(RunTest(229));
    ASSERT_TRUE(RunTest(230));
    ASSERT_TRUE(RunTest(231));
    ASSERT_TRUE(RunTest(232));
    ASSERT_TRUE(RunTest(233));
    ASSERT_TRUE(RunTest(234));
    ASSERT_TRUE(RunTest(235));
    ASSERT_TRUE(RunTest(236));
    ASSERT_TRUE(RunTest(237));
    ASSERT_TRUE(RunTest(238));
    ASSERT_TRUE(RunTest(239));
    ASSERT_TRUE(RunTest(240));
    ASSERT_TRUE(RunTest(241));
    ASSERT_TRUE(RunTest(242));
    ASSERT_TRUE(RunTest(243));
    ASSERT_TRUE(RunTest(244));
    ASSERT_TRUE(RunTest(245));
    ASSERT_TRUE(RunTest(246));
    ASSERT_TRUE(RunTest(247));
    ASSERT_TRUE(RunTest(248));
    ASSERT_TRUE(RunTest(249));
    ASSERT_TRUE(RunTest(250));
}
TEST(SBMLtestsuite, B)
{
    ASSERT_TRUE(RunTest(251));
    ASSERT_TRUE(RunTest(252));
    ASSERT_TRUE(RunTest(253));
    ASSERT_TRUE(RunTest(254));
    ASSERT_TRUE(RunTest(255));
    ASSERT_TRUE(RunTest(256));
    ASSERT_TRUE(RunTest(257));
    ASSERT_TRUE(RunTest(258));
    ASSERT_TRUE(RunTest(259));
    ASSERT_TRUE(RunTest(260));
    ASSERT_TRUE(RunTest(261));
    ASSERT_TRUE(RunTest(262));
    ASSERT_TRUE(RunTest(263));
    ASSERT_TRUE(RunTest(264));
    ASSERT_TRUE(RunTest(265));
    ASSERT_TRUE(RunTest(266));
    ASSERT_TRUE(RunTest(267));
    ASSERT_TRUE(RunTest(268));
    ASSERT_TRUE(RunTest(269));
    ASSERT_TRUE(RunTest(270));
    ASSERT_TRUE(RunTest(271));
    ASSERT_TRUE(RunTest(272));
    ASSERT_TRUE(RunTest(273));
    ASSERT_TRUE(RunTest(274));
    ASSERT_TRUE(RunTest(275));
    ASSERT_TRUE(RunTest(276));
    ASSERT_TRUE(RunTest(277));
    ASSERT_TRUE(RunTest(278));
    ASSERT_TRUE(RunTest(279));
    ASSERT_TRUE(RunTest(280));
    ASSERT_TRUE(RunTest(281));
    ASSERT_TRUE(RunTest(282));
    ASSERT_TRUE(RunTest(283));
    ASSERT_TRUE(RunTest(284));
    ASSERT_TRUE(RunTest(285));
    ASSERT_TRUE(RunTest(286));
    ASSERT_TRUE(RunTest(287));
    ASSERT_TRUE(RunTest(288));
    ASSERT_TRUE(RunTest(289));
    ASSERT_TRUE(RunTest(290));
    ASSERT_TRUE(RunTest(291));
    ASSERT_TRUE(RunTest(292));
    ASSERT_TRUE(RunTest(293));
    ASSERT_TRUE(RunTest(294));
    ASSERT_TRUE(RunTest(295));
    ASSERT_TRUE(RunTest(296));
    ASSERT_TRUE(RunTest(297));
    ASSERT_TRUE(RunTest(298));
    ASSERT_TRUE(RunTest(299));
    ASSERT_TRUE(RunTest(300));
    ASSERT_TRUE(RunTest(301));
    ASSERT_TRUE(RunTest(302));
    ASSERT_TRUE(RunTest(303));
    ASSERT_TRUE(RunTest(304));
    ASSERT_TRUE(RunTest(305));
    ASSERT_TRUE(RunTest(306));
    ASSERT_TRUE(RunTest(307));
    ASSERT_TRUE(RunTest(308));
    ASSERT_TRUE(RunTest(309));
    ASSERT_TRUE(RunTest(310));
    ASSERT_TRUE(RunTest(311));
    ASSERT_TRUE(RunTest(312));
    ASSERT_TRUE(RunTest(313));
    ASSERT_TRUE(RunTest(314));
    ASSERT_TRUE(RunTest(315));
    ASSERT_TRUE(RunTest(316));
    ASSERT_TRUE(RunTest(317));
    ASSERT_TRUE(RunTest(318));
    ASSERT_TRUE(RunTest(319));
    ASSERT_TRUE(RunTest(320));
    ASSERT_TRUE(RunTest(321));
    ASSERT_TRUE(RunTest(322));
    ASSERT_TRUE(RunTest(323));
    ASSERT_TRUE(RunTest(324));
    ASSERT_TRUE(RunTest(325));
    ASSERT_TRUE(RunTest(326));
    ASSERT_TRUE(RunTest(327));
    ASSERT_TRUE(RunTest(328));
    ASSERT_TRUE(RunTest(329));
    ASSERT_TRUE(RunTest(330));
    ASSERT_TRUE(RunTest(331));
    ASSERT_TRUE(RunTest(332));
    ASSERT_TRUE(RunTest(333));
    ASSERT_TRUE(RunTest(334));
    ASSERT_TRUE(RunTest(335));
    ASSERT_TRUE(RunTest(336));
    ASSERT_TRUE(RunTest(337));
    ASSERT_TRUE(RunTest(338));
    ASSERT_TRUE(RunTest(339));
    ASSERT_TRUE(RunTest(340));
    ASSERT_TRUE(RunTest(341));
    ASSERT_TRUE(RunTest(342));
    ASSERT_TRUE(RunTest(343));
    ASSERT_TRUE(RunTest(344));
    ASSERT_TRUE(RunTest(345));
    ASSERT_TRUE(RunTest(346));
    ASSERT_TRUE(RunTest(347));
    ASSERT_TRUE(RunTest(348));
    ASSERT_TRUE(RunTest(349));
    ASSERT_TRUE(RunTest(350));
    ASSERT_TRUE(RunTest(351));
    ASSERT_TRUE(RunTest(352));
    ASSERT_TRUE(RunTest(353));
    ASSERT_TRUE(RunTest(354));
    ASSERT_TRUE(RunTest(355));
    ASSERT_TRUE(RunTest(356));
    ASSERT_TRUE(RunTest(357));
    ASSERT_TRUE(RunTest(358));
    ASSERT_TRUE(RunTest(359));
    ASSERT_TRUE(RunTest(360));
    ASSERT_TRUE(RunTest(361));
    ASSERT_TRUE(RunTest(362));
    ASSERT_TRUE(RunTest(363));
    ASSERT_TRUE(RunTest(364));
    ASSERT_TRUE(RunTest(365));
    ASSERT_TRUE(RunTest(366));
    ASSERT_TRUE(RunTest(367));
    ASSERT_TRUE(RunTest(368));
    ASSERT_TRUE(RunTest(369));
    ASSERT_TRUE(RunTest(370));
    ASSERT_TRUE(RunTest(371));
    ASSERT_TRUE(RunTest(372));
    ASSERT_TRUE(RunTest(373));
    ASSERT_TRUE(RunTest(374));
    ASSERT_TRUE(RunTest(375));
    ASSERT_TRUE(RunTest(376));
    ASSERT_TRUE(RunTest(377));
    ASSERT_TRUE(RunTest(378));
    ASSERT_TRUE(RunTest(379));
    ASSERT_TRUE(RunTest(380));
    ASSERT_TRUE(RunTest(381));
    ASSERT_TRUE(RunTest(382));
    ASSERT_TRUE(RunTest(383));
    ASSERT_TRUE(RunTest(384));
    ASSERT_TRUE(RunTest(385));
    ASSERT_TRUE(RunTest(386));
    ASSERT_TRUE(RunTest(387));
    ASSERT_TRUE(RunTest(388));
    ASSERT_TRUE(RunTest(389));
    ASSERT_TRUE(RunTest(390));
    ASSERT_TRUE(RunTest(391));
    ASSERT_TRUE(RunTest(392));
    ASSERT_TRUE(RunTest(393));
    ASSERT_TRUE(RunTest(394));
    ASSERT_TRUE(RunTest(395));
    ASSERT_TRUE(RunTest(396));
    ASSERT_TRUE(RunTest(397));
    ASSERT_TRUE(RunTest(398));
    ASSERT_TRUE(RunTest(399));
    ASSERT_TRUE(RunTest(400));
    ASSERT_TRUE(RunTest(401));
    ASSERT_TRUE(RunTest(402));
    ASSERT_TRUE(RunTest(403));
    ASSERT_TRUE(RunTest(404));
    ASSERT_TRUE(RunTest(405));
    ASSERT_TRUE(RunTest(406));
    ASSERT_TRUE(RunTest(407));
    ASSERT_TRUE(RunTest(408));
    ASSERT_TRUE(RunTest(409));
    ASSERT_TRUE(RunTest(410));
    ASSERT_TRUE(RunTest(411));
    ASSERT_TRUE(RunTest(412));
    ASSERT_TRUE(RunTest(413));
    ASSERT_TRUE(RunTest(414));
    ASSERT_TRUE(RunTest(415));
    ASSERT_TRUE(RunTest(416));
    ASSERT_TRUE(RunTest(417));
    ASSERT_TRUE(RunTest(418));
    ASSERT_TRUE(RunTest(419));
    ASSERT_TRUE(RunTest(420));
    ASSERT_TRUE(RunTest(421));
    ASSERT_TRUE(RunTest(422));
    ASSERT_TRUE(RunTest(423));
    ASSERT_TRUE(RunTest(424));
    ASSERT_TRUE(RunTest(425));
    ASSERT_TRUE(RunTest(426));
    ASSERT_TRUE(RunTest(427));
    ASSERT_TRUE(RunTest(428));
    ASSERT_TRUE(RunTest(429));
    ASSERT_TRUE(RunTest(430));
    ASSERT_TRUE(RunTest(431));
    ASSERT_TRUE(RunTest(432));
    ASSERT_TRUE(RunTest(433));
    ASSERT_TRUE(RunTest(434));
    ASSERT_TRUE(RunTest(435));
    ASSERT_TRUE(RunTest(436));
    ASSERT_TRUE(RunTest(437));
    ASSERT_TRUE(RunTest(438));
    ASSERT_TRUE(RunTest(439));
    ASSERT_TRUE(RunTest(440));
    ASSERT_TRUE(RunTest(441));
    ASSERT_TRUE(RunTest(442));
    ASSERT_TRUE(RunTest(443));
    ASSERT_TRUE(RunTest(444));
    ASSERT_TRUE(RunTest(445));
    ASSERT_TRUE(RunTest(446));
    ASSERT_TRUE(RunTest(447));
    ASSERT_TRUE(RunTest(448));
    ASSERT_TRUE(RunTest(449));
    ASSERT_TRUE(RunTest(450));
    ASSERT_TRUE(RunTest(451));
    ASSERT_TRUE(RunTest(452));
    ASSERT_TRUE(RunTest(453));
    ASSERT_TRUE(RunTest(454));
    ASSERT_TRUE(RunTest(455));
    ASSERT_TRUE(RunTest(456));
    ASSERT_TRUE(RunTest(457));
    ASSERT_TRUE(RunTest(458));
    ASSERT_TRUE(RunTest(459));
    ASSERT_TRUE(RunTest(460));
    ASSERT_TRUE(RunTest(461));
    ASSERT_TRUE(RunTest(462));
    ASSERT_TRUE(RunTest(463));
    ASSERT_TRUE(RunTest(464));
    ASSERT_TRUE(RunTest(465));
    ASSERT_TRUE(RunTest(466));
    ASSERT_TRUE(RunTest(467));
    ASSERT_TRUE(RunTest(468));
    ASSERT_TRUE(RunTest(469));
    ASSERT_TRUE(RunTest(470));
    ASSERT_TRUE(RunTest(471));
    ASSERT_TRUE(RunTest(472));
    ASSERT_TRUE(RunTest(473));
    ASSERT_TRUE(RunTest(474));
    ASSERT_TRUE(RunTest(475));
    ASSERT_TRUE(RunTest(476));
    ASSERT_TRUE(RunTest(477));
    ASSERT_TRUE(RunTest(478));
    ASSERT_TRUE(RunTest(479));
    ASSERT_TRUE(RunTest(480));
    ASSERT_TRUE(RunTest(481));
    ASSERT_TRUE(RunTest(482));
    ASSERT_TRUE(RunTest(483));
    ASSERT_TRUE(RunTest(484));
    ASSERT_TRUE(RunTest(485));
    ASSERT_TRUE(RunTest(486));
    ASSERT_TRUE(RunTest(487));
    ASSERT_TRUE(RunTest(488));
    ASSERT_TRUE(RunTest(489));
    ASSERT_TRUE(RunTest(490));
    ASSERT_TRUE(RunTest(491));
    ASSERT_TRUE(RunTest(492));
    ASSERT_TRUE(RunTest(493));
    ASSERT_TRUE(RunTest(494));
    ASSERT_TRUE(RunTest(495));
    ASSERT_TRUE(RunTest(496));
    ASSERT_TRUE(RunTest(497));
    ASSERT_TRUE(RunTest(498));
    ASSERT_TRUE(RunTest(499));
    ASSERT_TRUE(RunTest(500));
}
TEST(SBMLtestsuite, C)
{
    ASSERT_TRUE(RunTest(501));
    ASSERT_TRUE(RunTest(502));
    ASSERT_TRUE(RunTest(503));
    ASSERT_TRUE(RunTest(504));
    ASSERT_TRUE(RunTest(505));
    ASSERT_TRUE(RunTest(506));
    ASSERT_TRUE(RunTest(507));
    ASSERT_TRUE(RunTest(508));
    ASSERT_TRUE(RunTest(509));
    ASSERT_TRUE(RunTest(510));
    ASSERT_TRUE(RunTest(511));
    ASSERT_TRUE(RunTest(512));
    ASSERT_TRUE(RunTest(513));
    ASSERT_TRUE(RunTest(514));
    ASSERT_TRUE(RunTest(515));
    ASSERT_TRUE(RunTest(516));
    ASSERT_TRUE(RunTest(517));
    ASSERT_TRUE(RunTest(518));
    ASSERT_TRUE(RunTest(519));
    ASSERT_TRUE(RunTest(520));
    ASSERT_TRUE(RunTest(521));
    ASSERT_TRUE(RunTest(522));
    ASSERT_TRUE(RunTest(523));
    ASSERT_TRUE(RunTest(524));
    ASSERT_TRUE(RunTest(525));
    ASSERT_TRUE(RunTest(526));
    ASSERT_TRUE(RunTest(527));
    ASSERT_TRUE(RunTest(528));
    ASSERT_TRUE(RunTest(529));
    ASSERT_TRUE(RunTest(530));
    ASSERT_TRUE(RunTest(531));
    ASSERT_TRUE(RunTest(532));
//    ASSERT_TRUE(RunTest(533)); // Throws because uninitialized value
//    ASSERT_TRUE(RunTest(534)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(535));
//    ASSERT_TRUE(RunTest(536)); // Throws because uninitialized value
//    ASSERT_TRUE(RunTest(537)); // Throws because uninitialized value
//    ASSERT_TRUE(RunTest(538)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(539));
    ASSERT_TRUE(RunTest(540));
    ASSERT_TRUE(RunTest(541));
    ASSERT_TRUE(RunTest(542));
    ASSERT_TRUE(RunTest(543));
    ASSERT_TRUE(RunTest(544));
    ASSERT_TRUE(RunTest(545));
    ASSERT_TRUE(RunTest(546));
    ASSERT_TRUE(RunTest(547));
    ASSERT_TRUE(RunTest(548));
    ASSERT_TRUE(RunTest(549));
    ASSERT_TRUE(RunTest(550));
    ASSERT_TRUE(RunTest(551));
    ASSERT_TRUE(RunTest(552));
    ASSERT_TRUE(RunTest(553));
    ASSERT_TRUE(RunTest(554));
    ASSERT_TRUE(RunTest(555));
    ASSERT_TRUE(RunTest(556));
    ASSERT_TRUE(RunTest(557));
    ASSERT_TRUE(RunTest(558));
    ASSERT_TRUE(RunTest(559));
    ASSERT_TRUE(RunTest(560));
    ASSERT_TRUE(RunTest(561));
    ASSERT_TRUE(RunTest(562));
    ASSERT_TRUE(RunTest(563));
    ASSERT_TRUE(RunTest(564));
    ASSERT_TRUE(RunTest(565));
    ASSERT_TRUE(RunTest(566));
    ASSERT_TRUE(RunTest(567));
    ASSERT_TRUE(RunTest(568));
//    ASSERT_TRUE(RunTest(569)); // Throws because uninitialized value
//    ASSERT_TRUE(RunTest(570)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(571));
    ASSERT_TRUE(RunTest(572));
    ASSERT_TRUE(RunTest(573));
    ASSERT_TRUE(RunTest(574));
//    ASSERT_TRUE(RunTest(575)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(576));
    ASSERT_TRUE(RunTest(577));
    ASSERT_TRUE(RunTest(578));
    ASSERT_TRUE(RunTest(579));
    ASSERT_TRUE(RunTest(580));
    ASSERT_TRUE(RunTest(581));
    ASSERT_TRUE(RunTest(582));
    ASSERT_TRUE(RunTest(583));
    ASSERT_TRUE(RunTest(584));
    ASSERT_TRUE(RunTest(585));
    ASSERT_TRUE(RunTest(586));
    ASSERT_TRUE(RunTest(587));
    ASSERT_TRUE(RunTest(588));
    ASSERT_TRUE(RunTest(589));
    ASSERT_TRUE(RunTest(590));
    ASSERT_TRUE(RunTest(591));
    ASSERT_TRUE(RunTest(592));
    ASSERT_TRUE(RunTest(593));
    ASSERT_TRUE(RunTest(594));
    ASSERT_TRUE(RunTest(595));
    ASSERT_TRUE(RunTest(596));
    ASSERT_TRUE(RunTest(597));
    ASSERT_TRUE(RunTest(598));
    ASSERT_TRUE(RunTest(599));
    ASSERT_TRUE(RunTest(600));
    ASSERT_TRUE(RunTest(601));
    ASSERT_TRUE(RunTest(602));
    ASSERT_TRUE(RunTest(603));
    ASSERT_TRUE(RunTest(604));
    ASSERT_TRUE(RunTest(605));
    ASSERT_TRUE(RunTest(606));
    ASSERT_TRUE(RunTest(607));
    ASSERT_TRUE(RunTest(608));
    ASSERT_TRUE(RunTest(609));
    ASSERT_TRUE(RunTest(610));
    ASSERT_TRUE(RunTest(611));
    ASSERT_TRUE(RunTest(612));
    ASSERT_TRUE(RunTest(613));
    ASSERT_TRUE(RunTest(614));
    ASSERT_TRUE(RunTest(615));
    ASSERT_TRUE(RunTest(616));
    ASSERT_TRUE(RunTest(617));
    ASSERT_TRUE(RunTest(618));
    ASSERT_TRUE(RunTest(619));
    ASSERT_TRUE(RunTest(620));
    ASSERT_TRUE(RunTest(621));
    ASSERT_TRUE(RunTest(622));
    ASSERT_TRUE(RunTest(623));
    ASSERT_TRUE(RunTest(624));
    ASSERT_TRUE(RunTest(625));
    ASSERT_TRUE(RunTest(626));
    ASSERT_TRUE(RunTest(627));
    ASSERT_TRUE(RunTest(628));
    ASSERT_TRUE(RunTest(629));
    ASSERT_TRUE(RunTest(630));
    ASSERT_TRUE(RunTest(631));
    ASSERT_TRUE(RunTest(632));
    ASSERT_TRUE(RunTest(633));
    ASSERT_TRUE(RunTest(634));
    ASSERT_TRUE(RunTest(635));
    ASSERT_TRUE(RunTest(636));
    ASSERT_TRUE(RunTest(637));
    ASSERT_TRUE(RunTest(638));
    ASSERT_TRUE(RunTest(639));
    ASSERT_TRUE(RunTest(640));
    ASSERT_TRUE(RunTest(641));
    ASSERT_TRUE(RunTest(642));
    ASSERT_TRUE(RunTest(643));
    ASSERT_TRUE(RunTest(644));
    ASSERT_TRUE(RunTest(645));
    ASSERT_TRUE(RunTest(646));
    ASSERT_TRUE(RunTest(647));
    ASSERT_TRUE(RunTest(648));
    ASSERT_TRUE(RunTest(649));
    ASSERT_TRUE(RunTest(650));
    ASSERT_TRUE(RunTest(651));
    ASSERT_TRUE(RunTest(652));
    ASSERT_TRUE(RunTest(653));
    ASSERT_TRUE(RunTest(654));
    ASSERT_TRUE(RunTest(655));
    ASSERT_TRUE(RunTest(656));
    ASSERT_TRUE(RunTest(657));
    ASSERT_TRUE(RunTest(658));
    ASSERT_TRUE(RunTest(659));
    ASSERT_TRUE(RunTest(660));
    ASSERT_TRUE(RunTest(661));
    ASSERT_TRUE(RunTest(662));
//    ASSERT_TRUE(RunTest(663)); // Throws because uninitialized value
//    ASSERT_TRUE(RunTest(664)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(665));
    ASSERT_TRUE(RunTest(666));
    ASSERT_TRUE(RunTest(667));
    ASSERT_TRUE(RunTest(668));
    ASSERT_TRUE(RunTest(669));
    ASSERT_TRUE(RunTest(670));
    ASSERT_TRUE(RunTest(671));
    ASSERT_TRUE(RunTest(672));
    ASSERT_TRUE(RunTest(673));
    ASSERT_TRUE(RunTest(674));
    ASSERT_TRUE(RunTest(675));
    ASSERT_TRUE(RunTest(676));
    ASSERT_TRUE(RunTest(677));
    ASSERT_TRUE(RunTest(678));
    ASSERT_TRUE(RunTest(679));
    ASSERT_TRUE(RunTest(680));
    ASSERT_TRUE(RunTest(681));
    ASSERT_TRUE(RunTest(682));
    ASSERT_TRUE(RunTest(683));
    ASSERT_TRUE(RunTest(684));
    ASSERT_TRUE(RunTest(685));
    ASSERT_TRUE(RunTest(686));
    ASSERT_TRUE(RunTest(687));
    ASSERT_TRUE(RunTest(688));
    ASSERT_TRUE(RunTest(689));
    ASSERT_TRUE(RunTest(690));
    ASSERT_TRUE(RunTest(691));
    ASSERT_TRUE(RunTest(692));
    ASSERT_TRUE(RunTest(693));
    ASSERT_TRUE(RunTest(694));
    ASSERT_TRUE(RunTest(695));
    ASSERT_TRUE(RunTest(696));
    ASSERT_TRUE(RunTest(697));
    ASSERT_TRUE(RunTest(698));
    ASSERT_TRUE(RunTest(699));
    ASSERT_TRUE(RunTest(700));
    ASSERT_TRUE(RunTest(701));
    ASSERT_TRUE(RunTest(702));
    ASSERT_TRUE(RunTest(703));
    ASSERT_TRUE(RunTest(704));
    ASSERT_TRUE(RunTest(705));
    ASSERT_TRUE(RunTest(706));
    ASSERT_TRUE(RunTest(707));
    ASSERT_TRUE(RunTest(708));
    ASSERT_TRUE(RunTest(709));
    ASSERT_TRUE(RunTest(710));
    ASSERT_TRUE(RunTest(711));
    ASSERT_TRUE(RunTest(712));
    ASSERT_TRUE(RunTest(713));
    ASSERT_TRUE(RunTest(714));
    ASSERT_TRUE(RunTest(715));
    ASSERT_TRUE(RunTest(716));
    ASSERT_TRUE(RunTest(717));
    ASSERT_TRUE(RunTest(718));
    ASSERT_TRUE(RunTest(719));
    ASSERT_TRUE(RunTest(720));
    ASSERT_TRUE(RunTest(721));
    ASSERT_TRUE(RunTest(722));
    ASSERT_TRUE(RunTest(723));
    ASSERT_TRUE(RunTest(724));
    ASSERT_TRUE(RunTest(725));
    ASSERT_TRUE(RunTest(726));
    ASSERT_TRUE(RunTest(727));
    ASSERT_TRUE(RunTest(728));
    ASSERT_TRUE(RunTest(729));
    ASSERT_TRUE(RunTest(730));
    ASSERT_TRUE(RunTest(731));
    ASSERT_TRUE(RunTest(732));
    ASSERT_TRUE(RunTest(733));
    ASSERT_TRUE(RunTest(734));
    ASSERT_TRUE(RunTest(735));
    ASSERT_TRUE(RunTest(736));
    ASSERT_TRUE(RunTest(737));
    ASSERT_TRUE(RunTest(738));
    ASSERT_TRUE(RunTest(739));
    ASSERT_TRUE(RunTest(740));
    ASSERT_TRUE(RunTest(741));
    ASSERT_TRUE(RunTest(742));
    ASSERT_TRUE(RunTest(743));
    ASSERT_TRUE(RunTest(744));
    ASSERT_TRUE(RunTest(745));
    ASSERT_TRUE(RunTest(746));
    ASSERT_TRUE(RunTest(747));
    ASSERT_TRUE(RunTest(748));
    ASSERT_TRUE(RunTest(749));
    ASSERT_TRUE(RunTest(750));
}
TEST(SBMLtestsuite, D)
{
    ASSERT_TRUE(RunTest(751));
    ASSERT_TRUE(RunTest(752));
    ASSERT_TRUE(RunTest(753));
    ASSERT_TRUE(RunTest(754));
    ASSERT_TRUE(RunTest(755));
    ASSERT_TRUE(RunTest(756));
    ASSERT_TRUE(RunTest(757));
    ASSERT_TRUE(RunTest(758));
    ASSERT_TRUE(RunTest(759));
    ASSERT_TRUE(RunTest(760));
    ASSERT_TRUE(RunTest(761));
//    ASSERT_TRUE(RunTest(762)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(763));
    ASSERT_TRUE(RunTest(764));
    ASSERT_TRUE(RunTest(765));
    ASSERT_TRUE(RunTest(766));
    ASSERT_TRUE(RunTest(767));
    ASSERT_TRUE(RunTest(768));
    ASSERT_TRUE(RunTest(769));
    ASSERT_TRUE(RunTest(770));
    ASSERT_TRUE(RunTest(771));
    ASSERT_TRUE(RunTest(772));
    ASSERT_TRUE(RunTest(773));
    ASSERT_TRUE(RunTest(774));
    ASSERT_TRUE(RunTest(775));
    ASSERT_TRUE(RunTest(776));
//    ASSERT_TRUE(RunTest(777)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(778));
    ASSERT_TRUE(RunTest(779));
    ASSERT_TRUE(RunTest(780));
    ASSERT_TRUE(RunTest(781));
    ASSERT_TRUE(RunTest(782));
    ASSERT_TRUE(RunTest(783));
    ASSERT_TRUE(RunTest(784));
    ASSERT_TRUE(RunTest(785));
    ASSERT_TRUE(RunTest(786));
    ASSERT_TRUE(RunTest(787));
    ASSERT_TRUE(RunTest(788));
    ASSERT_TRUE(RunTest(789));
    ASSERT_TRUE(RunTest(790));
    ASSERT_TRUE(RunTest(791));
    ASSERT_TRUE(RunTest(792));
    ASSERT_TRUE(RunTest(793));
    ASSERT_TRUE(RunTest(794));
    ASSERT_TRUE(RunTest(795));
    ASSERT_TRUE(RunTest(796));
    ASSERT_TRUE(RunTest(797));
    ASSERT_TRUE(RunTest(798));
    ASSERT_TRUE(RunTest(799));
    ASSERT_TRUE(RunTest(800));
    ASSERT_TRUE(RunTest(801));
    ASSERT_TRUE(RunTest(802));
    ASSERT_TRUE(RunTest(803));
    ASSERT_TRUE(RunTest(804));
    ASSERT_TRUE(RunTest(805));
    ASSERT_TRUE(RunTest(806));
    ASSERT_TRUE(RunTest(807));
    ASSERT_TRUE(RunTest(808));
    ASSERT_TRUE(RunTest(809));
    ASSERT_TRUE(RunTest(810));
    ASSERT_TRUE(RunTest(811));
    ASSERT_TRUE(RunTest(812));
    ASSERT_TRUE(RunTest(813));
    ASSERT_TRUE(RunTest(814));
    ASSERT_TRUE(RunTest(815));
    ASSERT_TRUE(RunTest(816));
    ASSERT_TRUE(RunTest(817));
    ASSERT_TRUE(RunTest(818));
    ASSERT_TRUE(RunTest(819));
    ASSERT_TRUE(RunTest(820));
    ASSERT_TRUE(RunTest(821));
    ASSERT_TRUE(RunTest(822));
    ASSERT_TRUE(RunTest(823));
    ASSERT_TRUE(RunTest(824));
    ASSERT_TRUE(RunTest(825));
    ASSERT_TRUE(RunTest(826));
    ASSERT_TRUE(RunTest(827));
    ASSERT_TRUE(RunTest(828));
    ASSERT_TRUE(RunTest(829));
    ASSERT_TRUE(RunTest(830));
    ASSERT_TRUE(RunTest(831));
    ASSERT_TRUE(RunTest(832));
    ASSERT_TRUE(RunTest(833));
    ASSERT_TRUE(RunTest(834));
    ASSERT_TRUE(RunTest(835));
    ASSERT_TRUE(RunTest(836));
    ASSERT_TRUE(RunTest(837));
    ASSERT_TRUE(RunTest(838));
    ASSERT_TRUE(RunTest(839));
    ASSERT_TRUE(RunTest(840));
    ASSERT_TRUE(RunTest(841));
    ASSERT_TRUE(RunTest(842));
    ASSERT_TRUE(RunTest(843));
//    ASSERT_TRUE(RunTest(844)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(845));
    ASSERT_TRUE(RunTest(846));
    ASSERT_TRUE(RunTest(847));
    ASSERT_TRUE(RunTest(848));
    ASSERT_TRUE(RunTest(849));
    ASSERT_TRUE(RunTest(850));
    ASSERT_TRUE(RunTest(851));
    ASSERT_TRUE(RunTest(852));
    ASSERT_TRUE(RunTest(853));
    ASSERT_TRUE(RunTest(854));
    ASSERT_TRUE(RunTest(855));
    ASSERT_TRUE(RunTest(856));
    ASSERT_TRUE(RunTest(857));
    ASSERT_TRUE(RunTest(858));
    ASSERT_TRUE(RunTest(859));
    ASSERT_TRUE(RunTest(860));
    ASSERT_TRUE(RunTest(861));
    ASSERT_TRUE(RunTest(862));
    ASSERT_TRUE(RunTest(863));
    ASSERT_TRUE(RunTest(864));
    ASSERT_TRUE(RunTest(865));
    ASSERT_TRUE(RunTest(866));
    ASSERT_TRUE(RunTest(867));
    ASSERT_TRUE(RunTest(868));
    ASSERT_TRUE(RunTest(869));
    ASSERT_TRUE(RunTest(870));
    ASSERT_TRUE(RunTest(871));
    ASSERT_TRUE(RunTest(872));
    ASSERT_TRUE(RunTest(873));
    ASSERT_TRUE(RunTest(874));
    ASSERT_TRUE(RunTest(875));
    ASSERT_TRUE(RunTest(876));
    ASSERT_TRUE(RunTest(877));
    ASSERT_TRUE(RunTest(878));
    ASSERT_TRUE(RunTest(879));
    ASSERT_TRUE(RunTest(880));
    ASSERT_TRUE(RunTest(881));
    ASSERT_TRUE(RunTest(882));
    ASSERT_TRUE(RunTest(883));
    ASSERT_TRUE(RunTest(884));
    ASSERT_TRUE(RunTest(885));
    ASSERT_TRUE(RunTest(886));
    ASSERT_TRUE(RunTest(887));
    ASSERT_TRUE(RunTest(888));
    ASSERT_TRUE(RunTest(889));
    ASSERT_TRUE(RunTest(890));
    ASSERT_TRUE(RunTest(891));
    ASSERT_TRUE(RunTest(892));
    ASSERT_TRUE(RunTest(893));
    ASSERT_TRUE(RunTest(894));
    ASSERT_TRUE(RunTest(895));
    ASSERT_TRUE(RunTest(896));
    ASSERT_TRUE(RunTest(897));
    ASSERT_TRUE(RunTest(898));
    ASSERT_TRUE(RunTest(899));
    ASSERT_TRUE(RunTest(900));
    ASSERT_TRUE(RunTest(901));
    ASSERT_TRUE(RunTest(902));
    ASSERT_TRUE(RunTest(903));
    ASSERT_TRUE(RunTest(904));
    ASSERT_TRUE(RunTest(905));
    ASSERT_TRUE(RunTest(906));
    ASSERT_TRUE(RunTest(907));
    ASSERT_TRUE(RunTest(908));
    ASSERT_TRUE(RunTest(909));
    ASSERT_TRUE(RunTest(910));
    ASSERT_TRUE(RunTest(911));
    ASSERT_TRUE(RunTest(912));
    ASSERT_TRUE(RunTest(913));
    ASSERT_TRUE(RunTest(914));
    ASSERT_TRUE(RunTest(915));
    ASSERT_TRUE(RunTest(916));
    ASSERT_TRUE(RunTest(917));
    ASSERT_TRUE(RunTest(918));
    ASSERT_TRUE(RunTest(919));
    ASSERT_TRUE(RunTest(920));
    ASSERT_TRUE(RunTest(921));
    ASSERT_TRUE(RunTest(922));
    ASSERT_TRUE(RunTest(923));
    ASSERT_TRUE(RunTest(924));
    ASSERT_TRUE(RunTest(925));
    ASSERT_TRUE(RunTest(926));
    ASSERT_TRUE(RunTest(927));
    ASSERT_TRUE(RunTest(928));
    ASSERT_TRUE(RunTest(929));
    ASSERT_TRUE(RunTest(930));
    ASSERT_TRUE(RunTest(931));
    ASSERT_TRUE(RunTest(932));
    ASSERT_TRUE(RunTest(933));
    ASSERT_TRUE(RunTest(934));
    ASSERT_TRUE(RunTest(935));
    ASSERT_TRUE(RunTest(936));
    ASSERT_TRUE(RunTest(937));
    ASSERT_TRUE(RunTest(938));
    ASSERT_TRUE(RunTest(939));
    ASSERT_TRUE(RunTest(940));
    ASSERT_TRUE(RunTest(941));
    ASSERT_TRUE(RunTest(942));
    ASSERT_TRUE(RunTest(943));
    ASSERT_TRUE(RunTest(944));
    ASSERT_TRUE(RunTest(945));
    ASSERT_TRUE(RunTest(946));
    ASSERT_TRUE(RunTest(947));
    ASSERT_TRUE(RunTest(948));
    ASSERT_TRUE(RunTest(949));
    ASSERT_TRUE(RunTest(950));
    ASSERT_TRUE(RunTest(951));
    ASSERT_TRUE(RunTest(952));
    ASSERT_TRUE(RunTest(953));
    ASSERT_TRUE(RunTest(954));
    ASSERT_TRUE(RunTest(955));
    ASSERT_TRUE(RunTest(956));
    ASSERT_TRUE(RunTest(957));
    ASSERT_TRUE(RunTest(958));
    ASSERT_TRUE(RunTest(959));
    ASSERT_TRUE(RunTest(960));
    ASSERT_TRUE(RunTest(961));
    ASSERT_TRUE(RunTest(962));
    ASSERT_TRUE(RunTest(963));
    ASSERT_TRUE(RunTest(964));
    ASSERT_TRUE(RunTest(965));
    ASSERT_TRUE(RunTest(966));
    ASSERT_TRUE(RunTest(967));
    ASSERT_TRUE(RunTest(968));
    ASSERT_TRUE(RunTest(969));
    ASSERT_TRUE(RunTest(970));
    ASSERT_TRUE(RunTest(971));
    ASSERT_TRUE(RunTest(972));
    ASSERT_TRUE(RunTest(973));
    ASSERT_TRUE(RunTest(974));
    ASSERT_TRUE(RunTest(975));
    ASSERT_TRUE(RunTest(976));
    ASSERT_TRUE(RunTest(977));
    ASSERT_TRUE(RunTest(978));
    ASSERT_TRUE(RunTest(979));
    ASSERT_TRUE(RunTest(980));
    ASSERT_TRUE(RunTest(981));
    ASSERT_TRUE(RunTest(982));
//    ASSERT_TRUE(RunTest(983)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(984));
    ASSERT_TRUE(RunTest(985));
    ASSERT_TRUE(RunTest(986));
    ASSERT_TRUE(RunTest(987));
    ASSERT_TRUE(RunTest(988));
    ASSERT_TRUE(RunTest(989));
    ASSERT_TRUE(RunTest(990));
    ASSERT_TRUE(RunTest(991));
    ASSERT_TRUE(RunTest(992));
//    ASSERT_TRUE(RunTest(993)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(994));
    ASSERT_TRUE(RunTest(995));
    ASSERT_TRUE(RunTest(996));
    ASSERT_TRUE(RunTest(997));
    ASSERT_TRUE(RunTest(998));
    ASSERT_TRUE(RunTest(999));
}
TEST(SBMLtestsuite, E)
{
    ASSERT_TRUE(RunTest(1000));
    ASSERT_TRUE(RunTest(1001));
    ASSERT_TRUE(RunTest(1002));
    ASSERT_TRUE(RunTest(1003));
    ASSERT_TRUE(RunTest(1004));
    ASSERT_TRUE(RunTest(1005));
    ASSERT_TRUE(RunTest(1006));
    ASSERT_TRUE(RunTest(1007));
    ASSERT_TRUE(RunTest(1008));
    ASSERT_TRUE(RunTest(1009));
    ASSERT_TRUE(RunTest(1010));
    ASSERT_TRUE(RunTest(1011));
    ASSERT_TRUE(RunTest(1012));
    ASSERT_TRUE(RunTest(1013));
    ASSERT_TRUE(RunTest(1014));
    ASSERT_TRUE(RunTest(1015));
    ASSERT_TRUE(RunTest(1016));
    ASSERT_TRUE(RunTest(1017));
    ASSERT_TRUE(RunTest(1018));
    ASSERT_TRUE(RunTest(1019));
    ASSERT_TRUE(RunTest(1020));
    ASSERT_TRUE(RunTest(1021));
    ASSERT_TRUE(RunTest(1022));
    ASSERT_TRUE(RunTest(1023));
    ASSERT_TRUE(RunTest(1024));
    ASSERT_TRUE(RunTest(1025));
    ASSERT_TRUE(RunTest(1026));
    ASSERT_TRUE(RunTest(1027));
    ASSERT_TRUE(RunTest(1028));
    ASSERT_TRUE(RunTest(1029));
    ASSERT_TRUE(RunTest(1030));
    ASSERT_TRUE(RunTest(1031));
    ASSERT_TRUE(RunTest(1032));
    ASSERT_TRUE(RunTest(1033));
    ASSERT_TRUE(RunTest(1034));
    ASSERT_TRUE(RunTest(1035));
    ASSERT_TRUE(RunTest(1036));
    ASSERT_TRUE(RunTest(1037));
    ASSERT_TRUE(RunTest(1038));
    ASSERT_TRUE(RunTest(1039));
    ASSERT_TRUE(RunTest(1040));
    ASSERT_TRUE(RunTest(1041));
    ASSERT_TRUE(RunTest(1042));
    ASSERT_TRUE(RunTest(1043));
//    ASSERT_TRUE(RunTest(1044)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(1045));
    ASSERT_TRUE(RunTest(1046));
    ASSERT_TRUE(RunTest(1047));
    ASSERT_TRUE(RunTest(1048));
    ASSERT_TRUE(RunTest(1049));
    ASSERT_TRUE(RunTest(1050));
    ASSERT_TRUE(RunTest(1051));
    ASSERT_TRUE(RunTest(1052));
    ASSERT_TRUE(RunTest(1053));
    ASSERT_TRUE(RunTest(1054));
    ASSERT_TRUE(RunTest(1055));
    ASSERT_TRUE(RunTest(1056));
    ASSERT_TRUE(RunTest(1057));
    ASSERT_TRUE(RunTest(1058));
    ASSERT_TRUE(RunTest(1059));
    ASSERT_TRUE(RunTest(1060));
    ASSERT_TRUE(RunTest(1061));
    ASSERT_TRUE(RunTest(1062));
    ASSERT_TRUE(RunTest(1063));
    ASSERT_TRUE(RunTest(1064));
    ASSERT_TRUE(RunTest(1065));
    ASSERT_TRUE(RunTest(1066));
    ASSERT_TRUE(RunTest(1067));
    ASSERT_TRUE(RunTest(1068));
    ASSERT_TRUE(RunTest(1069));
    ASSERT_TRUE(RunTest(1070));
    ASSERT_TRUE(RunTest(1071));
    ASSERT_TRUE(RunTest(1072));
    ASSERT_TRUE(RunTest(1073));
    ASSERT_TRUE(RunTest(1074));
    ASSERT_TRUE(RunTest(1075));
    ASSERT_TRUE(RunTest(1076));
    ASSERT_TRUE(RunTest(1077));
    ASSERT_TRUE(RunTest(1078));
    ASSERT_TRUE(RunTest(1079));
    ASSERT_TRUE(RunTest(1080));
    ASSERT_TRUE(RunTest(1081));
    ASSERT_TRUE(RunTest(1082));
    ASSERT_TRUE(RunTest(1083));
    ASSERT_TRUE(RunTest(1084));
    ASSERT_TRUE(RunTest(1085));
    ASSERT_TRUE(RunTest(1086));
    ASSERT_TRUE(RunTest(1087));
    ASSERT_TRUE(RunTest(1088));
    ASSERT_TRUE(RunTest(1089));
    ASSERT_TRUE(RunTest(1090));
    ASSERT_TRUE(RunTest(1091));
    ASSERT_TRUE(RunTest(1092));
    ASSERT_TRUE(RunTest(1093));
    ASSERT_TRUE(RunTest(1094));
    ASSERT_TRUE(RunTest(1095));
    ASSERT_TRUE(RunTest(1096));
    ASSERT_TRUE(RunTest(1097));
    ASSERT_TRUE(RunTest(1098));
    ASSERT_TRUE(RunTest(1099));
    ASSERT_TRUE(RunTest(1100));
    ASSERT_TRUE(RunTest(1101));
    ASSERT_TRUE(RunTest(1102));
    ASSERT_TRUE(RunTest(1103));
    ASSERT_TRUE(RunTest(1104));
    ASSERT_TRUE(RunTest(1105));
    ASSERT_TRUE(RunTest(1106));
    ASSERT_TRUE(RunTest(1107));
//    ASSERT_TRUE(RunTest(1108)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(1109));
    ASSERT_TRUE(RunTest(1110));
    ASSERT_TRUE(RunTest(1111));
    ASSERT_TRUE(RunTest(1112));
    ASSERT_TRUE(RunTest(1113));
    ASSERT_TRUE(RunTest(1114));
    ASSERT_TRUE(RunTest(1115));
    ASSERT_TRUE(RunTest(1116));
    ASSERT_TRUE(RunTest(1117));
    ASSERT_TRUE(RunTest(1118));
    ASSERT_TRUE(RunTest(1119));
    ASSERT_TRUE(RunTest(1120));
    ASSERT_TRUE(RunTest(1121));
    ASSERT_TRUE(RunTest(1122));
    ASSERT_TRUE(RunTest(1123));
    ASSERT_TRUE(RunTest(1124));
    ASSERT_TRUE(RunTest(1125));
    ASSERT_TRUE(RunTest(1126));
    ASSERT_TRUE(RunTest(1127));
    ASSERT_TRUE(RunTest(1128));
    ASSERT_TRUE(RunTest(1129));
    ASSERT_TRUE(RunTest(1130));
    ASSERT_TRUE(RunTest(1131));
    ASSERT_TRUE(RunTest(1132));
    ASSERT_TRUE(RunTest(1133));
    ASSERT_TRUE(RunTest(1134));
    ASSERT_TRUE(RunTest(1135));
    ASSERT_TRUE(RunTest(1136));
    ASSERT_TRUE(RunTest(1137));
    ASSERT_TRUE(RunTest(1138));
    ASSERT_TRUE(RunTest(1139));
    ASSERT_TRUE(RunTest(1140));
    ASSERT_TRUE(RunTest(1141));
    ASSERT_TRUE(RunTest(1142));
    ASSERT_TRUE(RunTest(1143));
    ASSERT_TRUE(RunTest(1144));
    ASSERT_TRUE(RunTest(1145));
    ASSERT_TRUE(RunTest(1146));
    ASSERT_TRUE(RunTest(1147));
    ASSERT_TRUE(RunTest(1148));
    ASSERT_TRUE(RunTest(1149));
    ASSERT_TRUE(RunTest(1150));
    ASSERT_TRUE(RunTest(1151));
    ASSERT_TRUE(RunTest(1152));
    ASSERT_TRUE(RunTest(1153));
    ASSERT_TRUE(RunTest(1154));
    ASSERT_TRUE(RunTest(1155));
    ASSERT_TRUE(RunTest(1156));
    ASSERT_TRUE(RunTest(1157));
    ASSERT_TRUE(RunTest(1158));
    ASSERT_TRUE(RunTest(1159));
    ASSERT_TRUE(RunTest(1160));
    ASSERT_TRUE(RunTest(1161));
    ASSERT_TRUE(RunTest(1162));
    ASSERT_TRUE(RunTest(1163));
    ASSERT_TRUE(RunTest(1164));
//    ASSERT_TRUE(RunTest(1165)); //Fail due to l3v2 flattening
    ASSERT_TRUE(RunTest(1166));
//    ASSERT_TRUE(RunTest(1167)); //Fail due to l3v2 flattening
//    ASSERT_TRUE(RunTest(1168)); //Fail due to l3v2 flattening
    ASSERT_TRUE(RunTest(1169));
    ASSERT_TRUE(RunTest(1170));
    ASSERT_TRUE(RunTest(1171));
    ASSERT_TRUE(RunTest(1172));
    ASSERT_TRUE(RunTest(1173));
    ASSERT_TRUE(RunTest(1174));
    ASSERT_TRUE(RunTest(1175));
    ASSERT_TRUE(RunTest(1176));
    ASSERT_TRUE(RunTest(1177));
    ASSERT_TRUE(RunTest(1178));
    ASSERT_TRUE(RunTest(1179));
    ASSERT_TRUE(RunTest(1180));
    ASSERT_TRUE(RunTest(1181));
    ASSERT_TRUE(RunTest(1182));
    ASSERT_TRUE(RunTest(1183));
    ASSERT_TRUE(RunTest(1184));
    ASSERT_TRUE(RunTest(1185));
    ASSERT_TRUE(RunTest(1186));
    ASSERT_TRUE(RunTest(1187));
    ASSERT_TRUE(RunTest(1188));
    ASSERT_TRUE(RunTest(1189));
    ASSERT_TRUE(RunTest(1190));
    ASSERT_TRUE(RunTest(1191));
    ASSERT_TRUE(RunTest(1192));
    ASSERT_TRUE(RunTest(1193));
    ASSERT_TRUE(RunTest(1194));
    ASSERT_TRUE(RunTest(1195));
    ASSERT_TRUE(RunTest(1196));
    ASSERT_TRUE(RunTest(1197));
//    ASSERT_TRUE(RunTest(1198)); // ASTNode destructor throws error.
    ASSERT_TRUE(RunTest(1199));
    ASSERT_TRUE(RunTest(1200));
    ASSERT_TRUE(RunTest(1201));
    ASSERT_TRUE(RunTest(1202));
    ASSERT_TRUE(RunTest(1203));
    ASSERT_TRUE(RunTest(1204));
    ASSERT_TRUE(RunTest(1205));
    ASSERT_TRUE(RunTest(1206));
    ASSERT_TRUE(RunTest(1207));
//    ASSERT_TRUE(RunTest(1208)); // ASTNode destructor throws error.
    ASSERT_TRUE(RunTest(1209));
    ASSERT_TRUE(RunTest(1210));
    ASSERT_TRUE(RunTest(1211));
    ASSERT_TRUE(RunTest(1212));
    ASSERT_TRUE(RunTest(1213));
    ASSERT_TRUE(RunTest(1214));
    ASSERT_TRUE(RunTest(1215));
    ASSERT_TRUE(RunTest(1216));
    ASSERT_TRUE(RunTest(1217));
    ASSERT_TRUE(RunTest(1218));
    ASSERT_TRUE(RunTest(1219));
    ASSERT_TRUE(RunTest(1220));
    ASSERT_TRUE(RunTest(1221));
    ASSERT_TRUE(RunTest(1222));
    ASSERT_TRUE(RunTest(1223));
    ASSERT_TRUE(RunTest(1224));
    ASSERT_TRUE(RunTest(1225));
    ASSERT_TRUE(RunTest(1226));
    ASSERT_TRUE(RunTest(1227));
    ASSERT_TRUE(RunTest(1228));
    ASSERT_TRUE(RunTest(1229));
    ASSERT_TRUE(RunTest(1230));
    ASSERT_TRUE(RunTest(1231));
    ASSERT_TRUE(RunTest(1232));
    ASSERT_TRUE(RunTest(1233));
//    ASSERT_TRUE(RunTest(1234)); // Missing ASTNode throw.
//    ASSERT_TRUE(RunTest(1235)); // Missing ASTNode throw.
//    ASSERT_TRUE(RunTest(1236)); // Missing ASTNode throw.
    ASSERT_TRUE(RunTest(1237));
//    ASSERT_TRUE(RunTest(1238)); // Unknown throw.
//    ASSERT_TRUE(RunTest(1239)); // Missing ASTNode throw.
    ASSERT_TRUE(RunTest(1240));
//    ASSERT_TRUE(RunTest(1241)); // Unknown throw.
//    ASSERT_TRUE(RunTest(1242)); // Unknown throw.
    ASSERT_TRUE(RunTest(1243));
//    ASSERT_TRUE(RunTest(1244)); // Unknown throw.
    ASSERT_TRUE(RunTest(1245));
    ASSERT_TRUE(RunTest(1246));
    ASSERT_TRUE(RunTest(1247));
//    ASSERT_TRUE(RunTest(1248)); // Failure to load model
//    ASSERT_TRUE(RunTest(1249)); // Failure to load model
//    ASSERT_TRUE(RunTest(1250)); // Failure to load model
}
TEST(SBMLtestsuite, F)
{
//    ASSERT_TRUE(RunTest(1251)); // Failure to load model
//    ASSERT_TRUE(RunTest(1252)); // Failure to load model
//    ASSERT_TRUE(RunTest(1253)); // Failure to load model
//    ASSERT_TRUE(RunTest(1254)); // Failure to load model
//    ASSERT_TRUE(RunTest(1255)); // Failure to load model
//    ASSERT_TRUE(RunTest(1256)); // Failure to load model
//    ASSERT_TRUE(RunTest(1257)); // Unknown throw
//    ASSERT_TRUE(RunTest(1258)); // Failure to load model
//    ASSERT_TRUE(RunTest(1259)); // Failure to load model
//    ASSERT_TRUE(RunTest(1260)); // Failure to load model
//    ASSERT_TRUE(RunTest(1261)); // Failure to load model
//    ASSERT_TRUE(RunTest(1262)); // Failure to load model
//    ASSERT_TRUE(RunTest(1263)); // Failure to load model
//    ASSERT_TRUE(RunTest(1264)); // Failure to load model
//    ASSERT_TRUE(RunTest(1265)); // Failure to load model
//    ASSERT_TRUE(RunTest(1266)); // Failure to load model
//    ASSERT_TRUE(RunTest(1267)); // Failure to load model
//    ASSERT_TRUE(RunTest(1268)); // Unknown throw
//    ASSERT_TRUE(RunTest(1269)); // Failure to load model
//    ASSERT_TRUE(RunTest(1270)); // Failure to load model
    ASSERT_TRUE(RunTest(1271));
    ASSERT_TRUE(RunTest(1272));
    ASSERT_TRUE(RunTest(1273));
    ASSERT_TRUE(RunTest(1274));
    ASSERT_TRUE(RunTest(1275));
    ASSERT_TRUE(RunTest(1276));
    ASSERT_TRUE(RunTest(1277));
    ASSERT_TRUE(RunTest(1278));
    ASSERT_TRUE(RunTest(1279));
    ASSERT_TRUE(RunTest(1280));
    ASSERT_TRUE(RunTest(1281));
//    ASSERT_TRUE(RunTest(1282)); // Throw; bad pointer.
//    ASSERT_TRUE(RunTest(1283)); // Throw; bad pointer.
//    ASSERT_TRUE(RunTest(1284)); // Throw; invalid cast.
//    ASSERT_TRUE(RunTest(1285)); // Throw; bad pointer.
//    ASSERT_TRUE(RunTest(1286)); //Throw; corrupt generated function (!)
    ASSERT_TRUE(RunTest(1287));
//    ASSERT_TRUE(RunTest(1288)); // Throw; bad pointer.
//    ASSERT_TRUE(RunTest(1289)); // Throw; bad pointer.
//    ASSERT_TRUE(RunTest(1290)); // Throw; bad pointer.
//    ASSERT_TRUE(RunTest(1291)); //Throw; corrupt generated function (!)
//    ASSERT_TRUE(RunTest(1292)); // Throws because uninitialized value
//    ASSERT_TRUE(RunTest(1293)); //Unknown throw
//    ASSERT_TRUE(RunTest(1294)); // Failure to load model
//    ASSERT_TRUE(RunTest(1295)); // Failure to load model
//    ASSERT_TRUE(RunTest(1296)); // Failure to load model
//    ASSERT_TRUE(RunTest(1297)); // Failure to load model
//    ASSERT_TRUE(RunTest(1298)); // Failure to load model
//    ASSERT_TRUE(RunTest(1299)); // Unknown throw
    ASSERT_TRUE(RunTest(1300));
    ASSERT_TRUE(RunTest(1301));
    ASSERT_TRUE(RunTest(1302));
    ASSERT_TRUE(RunTest(1303));
    ASSERT_TRUE(RunTest(1304));
    ASSERT_TRUE(RunTest(1305));
    ASSERT_TRUE(RunTest(1306));
    ASSERT_TRUE(RunTest(1307));
    ASSERT_TRUE(RunTest(1308));
    ASSERT_TRUE(RunTest(1309));
    ASSERT_TRUE(RunTest(1310));
    ASSERT_TRUE(RunTest(1311));
    ASSERT_TRUE(RunTest(1312));
    ASSERT_TRUE(RunTest(1313));
    ASSERT_TRUE(RunTest(1314));
    ASSERT_TRUE(RunTest(1315));
    ASSERT_TRUE(RunTest(1316));
    ASSERT_TRUE(RunTest(1317));
    ASSERT_TRUE(RunTest(1318));
    ASSERT_TRUE(RunTest(1319));
    ASSERT_TRUE(RunTest(1320));
//    ASSERT_TRUE(RunTest(1321)); // Failure to load model
//    ASSERT_TRUE(RunTest(1322)); // Failure to load model
    ASSERT_TRUE(RunTest(1323));
    ASSERT_TRUE(RunTest(1324));
    ASSERT_TRUE(RunTest(1325));
    ASSERT_TRUE(RunTest(1326));
    ASSERT_TRUE(RunTest(1327));
    ASSERT_TRUE(RunTest(1328));
    ASSERT_TRUE(RunTest(1329));
    ASSERT_TRUE(RunTest(1330));
    ASSERT_TRUE(RunTest(1331));
    ASSERT_TRUE(RunTest(1332));
    ASSERT_TRUE(RunTest(1333));
    ASSERT_TRUE(RunTest(1334));
    ASSERT_TRUE(RunTest(1335));
    ASSERT_TRUE(RunTest(1336));
    ASSERT_TRUE(RunTest(1337));
    ASSERT_TRUE(RunTest(1338));
    ASSERT_TRUE(RunTest(1339));
    ASSERT_TRUE(RunTest(1340));
    ASSERT_TRUE(RunTest(1341));
    ASSERT_TRUE(RunTest(1342));
    ASSERT_TRUE(RunTest(1343));
    ASSERT_TRUE(RunTest(1344));
    ASSERT_TRUE(RunTest(1345));
    ASSERT_TRUE(RunTest(1346));
    ASSERT_TRUE(RunTest(1347));
    ASSERT_TRUE(RunTest(1348));
    ASSERT_TRUE(RunTest(1349));
//    ASSERT_TRUE(RunTest(1350)); //Fail due to l3v2 flattening
    ASSERT_TRUE(RunTest(1351));
    ASSERT_TRUE(RunTest(1352));
    ASSERT_TRUE(RunTest(1353));
    ASSERT_TRUE(RunTest(1354));
    ASSERT_TRUE(RunTest(1355));
    ASSERT_TRUE(RunTest(1356));
    ASSERT_TRUE(RunTest(1357));
    ASSERT_TRUE(RunTest(1358));
//    ASSERT_TRUE(RunTest(1359)); //Fail due to l3v2 flattening
    ASSERT_TRUE(RunTest(1360));
    ASSERT_TRUE(RunTest(1361));
    ASSERT_TRUE(RunTest(1362));
    ASSERT_TRUE(RunTest(1363));
    ASSERT_TRUE(RunTest(1364));
    ASSERT_TRUE(RunTest(1365));
    ASSERT_TRUE(RunTest(1366));
    ASSERT_TRUE(RunTest(1367));
//    ASSERT_TRUE(RunTest(1368)); //Fail due to l3v2 flattening
    ASSERT_TRUE(RunTest(1369));
    ASSERT_TRUE(RunTest(1370));
    ASSERT_TRUE(RunTest(1371));
    ASSERT_TRUE(RunTest(1372));
    ASSERT_TRUE(RunTest(1373));
    ASSERT_TRUE(RunTest(1374));
    ASSERT_TRUE(RunTest(1375));
    ASSERT_TRUE(RunTest(1376));
//    ASSERT_TRUE(RunTest(1377)); //Fail due to l3v2 flattening
    ASSERT_TRUE(RunTest(1378));
    ASSERT_TRUE(RunTest(1379));
    ASSERT_TRUE(RunTest(1380));
    ASSERT_TRUE(RunTest(1381));
    ASSERT_TRUE(RunTest(1382));
    ASSERT_TRUE(RunTest(1383));
    ASSERT_TRUE(RunTest(1384));
    ASSERT_TRUE(RunTest(1385));
//    ASSERT_TRUE(RunTest(1386)); //Fail due to l3v2 flattening
    ASSERT_TRUE(RunTest(1387));
    ASSERT_TRUE(RunTest(1388));
    ASSERT_TRUE(RunTest(1389));
    ASSERT_TRUE(RunTest(1390));
    ASSERT_TRUE(RunTest(1391));
    ASSERT_TRUE(RunTest(1392));
    ASSERT_TRUE(RunTest(1393));
    ASSERT_TRUE(RunTest(1394));
    ASSERT_TRUE(RunTest(1395));
    ASSERT_TRUE(RunTest(1396));
    ASSERT_TRUE(RunTest(1397));
//    ASSERT_TRUE(RunTest(1398)); // Unknown throw
    ASSERT_TRUE(RunTest(1399));
//    ASSERT_TRUE(RunTest(1400)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1401)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1402)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1403)); // Failure to load model.
    ASSERT_TRUE(RunTest(1404));
//    ASSERT_TRUE(RunTest(1405)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1406)); // Failure to load model.
    ASSERT_TRUE(RunTest(1407));
//    ASSERT_TRUE(RunTest(1408)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1409)); // Failure to load model.
    ASSERT_TRUE(RunTest(1410));
    ASSERT_TRUE(RunTest(1411));
    ASSERT_TRUE(RunTest(1412));
    ASSERT_TRUE(RunTest(1413));
    ASSERT_TRUE(RunTest(1414));
    ASSERT_TRUE(RunTest(1415));
////    ASSERT_TRUE(RunTest(1416)); //Unknown throw
    ASSERT_TRUE(RunTest(1417));
    ASSERT_TRUE(RunTest(1418));
//    ASSERT_TRUE(RunTest(1419)); //Unknown throw
    ASSERT_TRUE(RunTest(1420));
    ASSERT_TRUE(RunTest(1421));
    ASSERT_TRUE(RunTest(1422));
    ASSERT_TRUE(RunTest(1423));
    ASSERT_TRUE(RunTest(1424));
    ASSERT_TRUE(RunTest(1425));
    ASSERT_TRUE(RunTest(1426));
    ASSERT_TRUE(RunTest(1427));
    ASSERT_TRUE(RunTest(1428));
    ASSERT_TRUE(RunTest(1429));
    ASSERT_TRUE(RunTest(1430));
    ASSERT_TRUE(RunTest(1431));
    ASSERT_TRUE(RunTest(1432));
    ASSERT_TRUE(RunTest(1433));
    ASSERT_TRUE(RunTest(1434));
    ASSERT_TRUE(RunTest(1435));
    ASSERT_TRUE(RunTest(1436));
    ASSERT_TRUE(RunTest(1437));
    ASSERT_TRUE(RunTest(1438));
    ASSERT_TRUE(RunTest(1439));
    ASSERT_TRUE(RunTest(1440));
    ASSERT_TRUE(RunTest(1441));
    ASSERT_TRUE(RunTest(1442));
    ASSERT_TRUE(RunTest(1443));
//    ASSERT_TRUE(RunTest(1444)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1445)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1446)); // Incorrect interpretation (species references)
//    ASSERT_TRUE(RunTest(1447)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1448)); // Unknown throw
//    ASSERT_TRUE(RunTest(1449)); // Unknown throw
//    ASSERT_TRUE(RunTest(1450)); // Unknown throw
//    ASSERT_TRUE(RunTest(1451)); // Incorrect interpretation
//    ASSERT_TRUE(RunTest(1452)); // Unknown throw
//    ASSERT_TRUE(RunTest(1453)); // Unknown throw
//    ASSERT_TRUE(RunTest(1454)); // Unknown throw
//    ASSERT_TRUE(RunTest(1455)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1456)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1457)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1458)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1459)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1460)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1461)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1462)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1463)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1464)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1465)); // Unknown throw.
    ASSERT_TRUE(RunTest(1466));
    ASSERT_TRUE(RunTest(1467));
    ASSERT_TRUE(RunTest(1468));
    ASSERT_TRUE(RunTest(1469));
    ASSERT_TRUE(RunTest(1470));
//    ASSERT_TRUE(RunTest(1471)); //Fail due to l3v2 flattening
//    ASSERT_TRUE(RunTest(1472)); //Fail due to l3v2 flattening
//    ASSERT_TRUE(RunTest(1473)); //Fail due to l3v2 flattening
    ASSERT_TRUE(RunTest(1474));
//    ASSERT_TRUE(RunTest(1475)); //Fail due to l3v2 flattening
//    ASSERT_TRUE(RunTest(1476)); //Fail due to l3v2 flattening
//    ASSERT_TRUE(RunTest(1477)); //Fail due to l3v2 flattening
//    ASSERT_TRUE(RunTest(1478)); // Unknown throw.
//    ASSERT_TRUE(RunTest(1479)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(1480));
    ASSERT_TRUE(RunTest(1481));
//    ASSERT_TRUE(RunTest(1482)); // Throws because uninitialized value
//    ASSERT_TRUE(RunTest(1483)); // Throws because uninitialized value
//    ASSERT_TRUE(RunTest(1484)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(1485));
    ASSERT_TRUE(RunTest(1486));
    ASSERT_TRUE(RunTest(1487));
//    ASSERT_TRUE(RunTest(1488)); // Incorrect interpretation: odd mathml in functions
    ASSERT_TRUE(RunTest(1489));
    ASSERT_TRUE(RunTest(1490));
    ASSERT_TRUE(RunTest(1491));
    ASSERT_TRUE(RunTest(1492));
    ASSERT_TRUE(RunTest(1493));
    ASSERT_TRUE(RunTest(1494));
    ASSERT_TRUE(RunTest(1495));
    ASSERT_TRUE(RunTest(1496));
    ASSERT_TRUE(RunTest(1497));
//    ASSERT_TRUE(RunTest(1498)); // Unknown throw.
//    ASSERT_TRUE(RunTest(1499)); // Throws because uninitialized value
//    ASSERT_TRUE(RunTest(1500)); // Throws because uninitialized value
}
TEST(SBMLtestsuite, G)
{
//    ASSERT_TRUE(RunTest(1501)); // Throws because uninitialized value
//    ASSERT_TRUE(RunTest(1502)); // Throws because uninitialized value
//    ASSERT_TRUE(RunTest(1503)); // Throws because uninitialized value
//    ASSERT_TRUE(RunTest(1504)); // Unknown throw
//    ASSERT_TRUE(RunTest(1505)); // Unknown throw
//    ASSERT_TRUE(RunTest(1506)); // Unknown throw
//    ASSERT_TRUE(RunTest(1507)); // Unknown throw
//    ASSERT_TRUE(RunTest(1508)); // Unknown throw
//    ASSERT_TRUE(RunTest(1509)); // Unknown throw
//    ASSERT_TRUE(RunTest(1510)); // Unknown throw
//    ASSERT_TRUE(RunTest(1511)); // Unknown throw
//    ASSERT_TRUE(RunTest(1512)); // Unknown throw
//    ASSERT_TRUE(RunTest(1513)); // Unknown throw
//    ASSERT_TRUE(RunTest(1514)); // Unknown throw
    ASSERT_TRUE(RunTest(1515));
    ASSERT_TRUE(RunTest(1516));
//    ASSERT_TRUE(RunTest(1517)); // Incorrect mathml interpretation
    ASSERT_TRUE(RunTest(1518));
    ASSERT_TRUE(RunTest(1519));
    ASSERT_TRUE(RunTest(1520));
    ASSERT_TRUE(RunTest(1521));
    ASSERT_TRUE(RunTest(1522));
    ASSERT_TRUE(RunTest(1523));
    ASSERT_TRUE(RunTest(1524));
//    ASSERT_TRUE(RunTest(1525)); // Unknown throw
//    ASSERT_TRUE(RunTest(1526)); // Unknown throw
//    ASSERT_TRUE(RunTest(1527)); // Unknown throw
//    ASSERT_TRUE(RunTest(1528)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1529)); // Failure to load model.
    ASSERT_TRUE(RunTest(1530));
    ASSERT_TRUE(RunTest(1531));
    ASSERT_TRUE(RunTest(1532));
    ASSERT_TRUE(RunTest(1533));
    ASSERT_TRUE(RunTest(1534));
    ASSERT_TRUE(RunTest(1535));
    ASSERT_TRUE(RunTest(1536));
    ASSERT_TRUE(RunTest(1537));
    ASSERT_TRUE(RunTest(1538));
    ASSERT_TRUE(RunTest(1539));
//    ASSERT_TRUE(RunTest(1540)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1541)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1542)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1543)); // Failure to load model.
    ASSERT_TRUE(RunTest(1544));
    ASSERT_TRUE(RunTest(1545));
    ASSERT_TRUE(RunTest(1546));
    ASSERT_TRUE(RunTest(1547));
    ASSERT_TRUE(RunTest(1548));
    ASSERT_TRUE(RunTest(1549));
    ASSERT_TRUE(RunTest(1550));
    ASSERT_TRUE(RunTest(1551));
//    ASSERT_TRUE(RunTest(1552)); // Unknown throw.
//    ASSERT_TRUE(RunTest(1553)); // Unknown throw.
//    ASSERT_TRUE(RunTest(1554)); // Unknown throw.
    ASSERT_TRUE(RunTest(1555));
//    ASSERT_TRUE(RunTest(1556)); // Unknown throw.
    ASSERT_TRUE(RunTest(1557));
    ASSERT_TRUE(RunTest(1558));
    ASSERT_TRUE(RunTest(1559));
    ASSERT_TRUE(RunTest(1560));
    ASSERT_TRUE(RunTest(1561));
    ASSERT_TRUE(RunTest(1562));
    ASSERT_TRUE(RunTest(1563));
    ASSERT_TRUE(RunTest(1564));
    ASSERT_TRUE(RunTest(1565));
    ASSERT_TRUE(RunTest(1566));
//    ASSERT_TRUE(RunTest(1567)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(1568));
    ASSERT_TRUE(RunTest(1569));
    ASSERT_TRUE(RunTest(1570));
//    ASSERT_TRUE(RunTest(1571)); // Unknown throw.
    ASSERT_TRUE(RunTest(1572));
    ASSERT_TRUE(RunTest(1573));
//    ASSERT_TRUE(RunTest(1574)); // Unknown throw.
//    ASSERT_TRUE(RunTest(1575)); // Throws because uninitialized value
//    ASSERT_TRUE(RunTest(1576)); // Throws because uninitialized value
//    ASSERT_TRUE(RunTest(1577)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(1578));
//    ASSERT_TRUE(RunTest(1579)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(1580));
    ASSERT_TRUE(RunTest(1581));
    ASSERT_TRUE(RunTest(1582));
    ASSERT_TRUE(RunTest(1583));
    ASSERT_TRUE(RunTest(1584));
    ASSERT_TRUE(RunTest(1585));
    ASSERT_TRUE(RunTest(1586));
    ASSERT_TRUE(RunTest(1587));
    ASSERT_TRUE(RunTest(1588));
//    ASSERT_TRUE(RunTest(1589)); // Throws because uninitialized value
    ASSERT_TRUE(RunTest(1590));
    ASSERT_TRUE(RunTest(1591));
    ASSERT_TRUE(RunTest(1592));
    ASSERT_TRUE(RunTest(1593));
    ASSERT_TRUE(RunTest(1594));
    ASSERT_TRUE(RunTest(1595));
    ASSERT_TRUE(RunTest(1596));
    ASSERT_TRUE(RunTest(1597));
    ASSERT_TRUE(RunTest(1598));
    ASSERT_TRUE(RunTest(1599));
    ASSERT_TRUE(RunTest(1600));
    ASSERT_TRUE(RunTest(1601));
    ASSERT_TRUE(RunTest(1602));
    ASSERT_TRUE(RunTest(1603));
    ASSERT_TRUE(RunTest(1604));
    ASSERT_TRUE(RunTest(1605));
    ASSERT_TRUE(RunTest(1606));
    ASSERT_TRUE(RunTest(1607));
    ASSERT_TRUE(RunTest(1608));
    ASSERT_TRUE(RunTest(1609));
    ASSERT_TRUE(RunTest(1610));
    ASSERT_TRUE(RunTest(1611));
    ASSERT_TRUE(RunTest(1612));
    ASSERT_TRUE(RunTest(1613));
    ASSERT_TRUE(RunTest(1614));
    ASSERT_TRUE(RunTest(1615));
    ASSERT_TRUE(RunTest(1616));
    ASSERT_TRUE(RunTest(1617));
    ASSERT_TRUE(RunTest(1618));
    ASSERT_TRUE(RunTest(1619));
    ASSERT_TRUE(RunTest(1620));
    ASSERT_TRUE(RunTest(1621));
    ASSERT_TRUE(RunTest(1622));
    ASSERT_TRUE(RunTest(1623));
    ASSERT_TRUE(RunTest(1624));
    ASSERT_TRUE(RunTest(1625));
//    ASSERT_TRUE(RunTest(1626)); // Unknown throw.
//    ASSERT_TRUE(RunTest(1627)); // Unknown throw.
//    ASSERT_TRUE(RunTest(1628)); // Throw; AST unknown pointer.
//    ASSERT_TRUE(RunTest(1629)); // Throw; AST unknown pointer.
//    ASSERT_TRUE(RunTest(1630)); // Throw; AST unknown pointer.
    ASSERT_TRUE(RunTest(1631));
    ASSERT_TRUE(RunTest(1632));
    ASSERT_TRUE(RunTest(1633));
    ASSERT_TRUE(RunTest(1634));
//    ASSERT_TRUE(RunTest(1635)); // Unknown throw.
//    ASSERT_TRUE(RunTest(1636)); // l2v1 passes, l2v5 throws (!)
    ASSERT_TRUE(RunTest(1637));
    ASSERT_TRUE(RunTest(1638));
    ASSERT_TRUE(RunTest(1639));
    ASSERT_TRUE(RunTest(1640));
    ASSERT_TRUE(RunTest(1641));
    ASSERT_TRUE(RunTest(1642));
    ASSERT_TRUE(RunTest(1643));
    ASSERT_TRUE(RunTest(1644));
    ASSERT_TRUE(RunTest(1645));
    ASSERT_TRUE(RunTest(1646));
    ASSERT_TRUE(RunTest(1647));
    ASSERT_TRUE(RunTest(1648));
    ASSERT_TRUE(RunTest(1649));
    ASSERT_TRUE(RunTest(1650));
    ASSERT_TRUE(RunTest(1651));
    ASSERT_TRUE(RunTest(1652));
    ASSERT_TRUE(RunTest(1653));
    ASSERT_TRUE(RunTest(1654));
    ASSERT_TRUE(RunTest(1655));
    ASSERT_TRUE(RunTest(1656));
//    ASSERT_TRUE(RunTest(1657)); // Throw; AST unknown pointer.
    ASSERT_TRUE(RunTest(1658));
    ASSERT_TRUE(RunTest(1659));
    ASSERT_TRUE(RunTest(1660));
    ASSERT_TRUE(RunTest(1661));
    ASSERT_TRUE(RunTest(1662));
    ASSERT_TRUE(RunTest(1663));
    ASSERT_TRUE(RunTest(1664));
    ASSERT_TRUE(RunTest(1665));
    ASSERT_TRUE(RunTest(1666));
    ASSERT_TRUE(RunTest(1667));
    ASSERT_TRUE(RunTest(1668));
    ASSERT_TRUE(RunTest(1669));
    ASSERT_TRUE(RunTest(1670));
    ASSERT_TRUE(RunTest(1671));
    ASSERT_TRUE(RunTest(1672));
    ASSERT_TRUE(RunTest(1673));
    ASSERT_TRUE(RunTest(1674));
    ASSERT_TRUE(RunTest(1675));
    ASSERT_TRUE(RunTest(1676));
    ASSERT_TRUE(RunTest(1677));
    ASSERT_TRUE(RunTest(1678));
    ASSERT_TRUE(RunTest(1679));
    ASSERT_TRUE(RunTest(1680));
    ASSERT_TRUE(RunTest(1681));
    ASSERT_TRUE(RunTest(1682));
    ASSERT_TRUE(RunTest(1683));
    ASSERT_TRUE(RunTest(1684));
    ASSERT_TRUE(RunTest(1685));
    ASSERT_TRUE(RunTest(1686));
    ASSERT_TRUE(RunTest(1687));
    ASSERT_TRUE(RunTest(1688));
    ASSERT_TRUE(RunTest(1689));
    ASSERT_TRUE(RunTest(1690));
    ASSERT_TRUE(RunTest(1691));
    ASSERT_TRUE(RunTest(1692));
    ASSERT_TRUE(RunTest(1693));
    ASSERT_TRUE(RunTest(1694));
    ASSERT_TRUE(RunTest(1695));
    ASSERT_TRUE(RunTest(1696));
    ASSERT_TRUE(RunTest(1697));
    ASSERT_TRUE(RunTest(1698));
    ASSERT_TRUE(RunTest(1699));
    ASSERT_TRUE(RunTest(1700));
    ASSERT_TRUE(RunTest(1701));
    ASSERT_TRUE(RunTest(1702));
    ASSERT_TRUE(RunTest(1703));
    ASSERT_TRUE(RunTest(1704));
    ASSERT_TRUE(RunTest(1705));
    ASSERT_TRUE(RunTest(1706));
    ASSERT_TRUE(RunTest(1707));
    ASSERT_TRUE(RunTest(1708));
    ASSERT_TRUE(RunTest(1709));
    ASSERT_TRUE(RunTest(1710));
    ASSERT_TRUE(RunTest(1711));
    ASSERT_TRUE(RunTest(1712));
    ASSERT_TRUE(RunTest(1713));
    ASSERT_TRUE(RunTest(1714));
    ASSERT_TRUE(RunTest(1715));
    ASSERT_TRUE(RunTest(1716));
    ASSERT_TRUE(RunTest(1717));
    ASSERT_TRUE(RunTest(1718));
    ASSERT_TRUE(RunTest(1719));
    ASSERT_TRUE(RunTest(1720));
    ASSERT_TRUE(RunTest(1721));
    ASSERT_TRUE(RunTest(1722));
    ASSERT_TRUE(RunTest(1723));
    ASSERT_TRUE(RunTest(1724));
    ASSERT_TRUE(RunTest(1725));
    ASSERT_TRUE(RunTest(1726));
//    ASSERT_TRUE(RunTest(1727)); // Unknown throw.
//    ASSERT_TRUE(RunTest(1728)); // Unknown throw.
//    ASSERT_TRUE(RunTest(1729)); // Unknown throw.
    ASSERT_TRUE(RunTest(1730));
    ASSERT_TRUE(RunTest(1731));
    ASSERT_TRUE(RunTest(1732));
    ASSERT_TRUE(RunTest(1733));
    ASSERT_TRUE(RunTest(1734));
    ASSERT_TRUE(RunTest(1735));
//    ASSERT_TRUE(RunTest(1736)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1737)); // Failure to load model.
//    ASSERT_TRUE(RunTest(1738)); // Failure to load model.
    ASSERT_TRUE(RunTest(1739));
    ASSERT_TRUE(RunTest(1740));
    ASSERT_TRUE(RunTest(1741));
    ASSERT_TRUE(RunTest(1742));
    ASSERT_TRUE(RunTest(1743));
    ASSERT_TRUE(RunTest(1744));
    ASSERT_TRUE(RunTest(1745));
    ASSERT_TRUE(RunTest(1746));
    ASSERT_TRUE(RunTest(1747));
    ASSERT_TRUE(RunTest(1748));
    ASSERT_TRUE(RunTest(1749));
    ASSERT_TRUE(RunTest(1750));
}
TEST(SBMLtestsuite, H)
{
    ASSERT_TRUE(RunTest(1751));
    ASSERT_TRUE(RunTest(1752));
    ASSERT_TRUE(RunTest(1753));
    ASSERT_TRUE(RunTest(1754));
    ASSERT_TRUE(RunTest(1755));
    ASSERT_TRUE(RunTest(1756));
    ASSERT_TRUE(RunTest(1757));
    ASSERT_TRUE(RunTest(1758));
    ASSERT_TRUE(RunTest(1759));
    ASSERT_TRUE(RunTest(1760));
    ASSERT_TRUE(RunTest(1761));
    ASSERT_TRUE(RunTest(1762));
    ASSERT_TRUE(RunTest(1763));
    ASSERT_TRUE(RunTest(1764));
    ASSERT_TRUE(RunTest(1765));
    ASSERT_TRUE(RunTest(1766));
    ASSERT_TRUE(RunTest(1767));
    ASSERT_TRUE(RunTest(1768));
    ASSERT_TRUE(RunTest(1769));
    ASSERT_TRUE(RunTest(1770));
    ASSERT_TRUE(RunTest(1771));
    ASSERT_TRUE(RunTest(1772));
    ASSERT_TRUE(RunTest(1773));
    ASSERT_TRUE(RunTest(1774));
    ASSERT_TRUE(RunTest(1775));
    ASSERT_TRUE(RunTest(1776));
    ASSERT_TRUE(RunTest(1777));
//    ASSERT_TRUE(RunTest(1778)); //Fail due to l3v2 flattening
    ASSERT_TRUE(RunTest(1779));
//    ASSERT_TRUE(RunTest(1780)); // Incorrect interpretation.
    ASSERT_TRUE(RunTest(1781));
    ASSERT_TRUE(RunTest(1782));
    ASSERT_TRUE(RunTest(1783));
    ASSERT_TRUE(RunTest(1784));
    ASSERT_TRUE(RunTest(1785));
    ASSERT_TRUE(RunTest(1786));
    ASSERT_TRUE(RunTest(1787));
    ASSERT_TRUE(RunTest(1788));
    ASSERT_TRUE(RunTest(1789));
    ASSERT_TRUE(RunTest(1790));
    ASSERT_TRUE(RunTest(1791));
    ASSERT_TRUE(RunTest(1792));
    ASSERT_TRUE(RunTest(1793));
    ASSERT_TRUE(RunTest(1794));
    ASSERT_TRUE(RunTest(1795));
    ASSERT_TRUE(RunTest(1796));
    ASSERT_TRUE(RunTest(1797));
    ASSERT_TRUE(RunTest(1798));
    ASSERT_TRUE(RunTest(1799));
    ASSERT_TRUE(RunTest(1800));
    ASSERT_TRUE(RunTest(1801));
    ASSERT_TRUE(RunTest(1802));
    ASSERT_TRUE(RunTest(1803));
    ASSERT_TRUE(RunTest(1804));
    ASSERT_TRUE(RunTest(1805));
    ASSERT_TRUE(RunTest(1806));
    ASSERT_TRUE(RunTest(1807));
    ASSERT_TRUE(RunTest(1808));
    ASSERT_TRUE(RunTest(1809));
}


bool RunTest(int caseNumber)
{
    //Run the first and last version of the file.
    string modelFileName, settingsFileName, descriptionFileName;
    vector<string> lvs;
    lvs.push_back("l1v2"); 
    lvs.push_back("l2v1"); 
    lvs.push_back("l2v2"); 
    lvs.push_back("l2v3"); 
    lvs.push_back("l2v4"); 
    lvs.push_back("l2v5"); 
    lvs.push_back("l3v1");
    lvs.push_back("l3v2");
    string testsuitedir = gRRTestDir + "sbml-test-suite/semantic/";
    string modelFilePath(testsuitedir);
    string first = "";
    string last = "";
    for (size_t n = 0; n < lvs.size(); n++) {
        string lv = lvs[n];
        modelFilePath = testsuitedir; //Reset, since the subdir is added.
        createTestSuiteFileNameParts(caseNumber, "-sbml-" + lv + ".xml", modelFilePath, modelFileName, settingsFileName, descriptionFileName);
        ifstream ftest(modelFilePath + "/" + modelFileName);
        if (ftest.good()) {
            if (first.empty()) {
                first = lv;
            }
            else {
                last = lv;
            }
        }
    }
    bool ret = true;
    if (hasUnimplementedTags(modelFilePath + "/" + descriptionFileName)) {
        if (!first.empty()) {
            ret = CheckLoad(first, caseNumber);
        }
        else {
            Log(Logger::LOG_ERROR) << "No models found for test case" << caseNumber << endl;
            ret = false;
        }
        if (!last.empty()) {
            ret = CheckLoad(last, caseNumber) && ret;
        }
    }

    else {
        if (!first.empty()) {
            ret = RunTest(first, caseNumber);
            if (!ret && isSemiStochasticTest(modelFilePath + "/" + descriptionFileName)) {
                //semistochastic tests fail once in a great while, but very very rarely twice in a row.
                Log(Logger::LOG_WARNING) << "Test " << caseNumber << " failed, but we expect it to fail every so often.  Trying again...";
                ret = RunTest(first, caseNumber);
            }
        }
        else {
            Log(Logger::LOG_ERROR) << "No models found for test case" << caseNumber << endl;
            ret = false;
        }
        if (!last.empty()) {
            ret = RunTest(last, caseNumber) && ret;
        }
    }
    return ret;
}


bool RunTest(const string& version, int caseNumber)
{
    cerr << "Running Test:\t" << caseNumber << "\t" << version;

    RoadRunner rr;
    TestSuiteModelSimulation simulation;
    try
    {
        LoadAndSimulate(version, caseNumber, rr, simulation);

        //Write result
        if(!simulation.SaveResult())
        {
            //Failed to save data
            Log(Logger::LOG_ERROR)<<"Failed to save result";
            throw("Failed running simulation: Failed to save result");
        }

        if(!simulation.LoadReferenceData())
        {
            Log(Logger::LOG_ERROR)<<"Failed loading reference data";
            throw("Failed loading reference data");
        }

        simulation.CreateErrorData();
        bool result = simulation.Pass();
        simulation.SaveAllData();

        //simulation.SaveModelAsXML(dataOutputFolder);

        cerr<<"\t"<< (result == true ? "PASS" : "FAIL")<<endl;
        return result;
     }
    catch(Exception& ex)
    {
        string error = ex.what();
        cerr << "\tFAIL" << endl;
        cerr<<"Case "<<caseNumber<<": Exception: "<<error<<endl;
        return false;
    }

}

bool CheckLoad(const string& version, int caseNumber)
{
    cerr << "Checking Test Loading:\t" << caseNumber << "\t" << version;

    RoadRunner rr;
    TestSuiteModelSimulation simulation;

    try
    {
        LoadAndSimulate(version, caseNumber, rr, simulation);

        //If we've gotten this far, rejoice!  roadrunner didn't crash, which is good enough.
        cerr << "\tPASS" << endl;
        return true;
    }
    catch (Exception& ex)
    {
        string error = ex.what();
        cerr << "\tFAIL" << endl;
        cerr << "Case " << caseNumber << ": Exception: " << error << endl;
        return false;
    }

}

void LoadAndSimulate(const string& version, int caseNumber, RoadRunner& rr, TestSuiteModelSimulation& simulation)
{

    string dummy;
    string logFileName;

    rr.getIntegrator()->setValue("stiff", false);

    //Create log file name, e.g. 00001.log
    createTestSuiteFileNameParts(caseNumber, "_" + version + ".log", dummy, logFileName, dummy, dummy);

    //rr.reset();
    simulation.UseEngine(&rr);

    //Setup filenames and paths...
    string testsuitedir = gRRTestDir + "sbml-test-suite/semantic/";
    string modelFilePath(testsuitedir);
    string modelFileName;
    string settingsFileName;
    string descriptionFileName;
    createTestSuiteFileNameParts(caseNumber, "-sbml-" + version + ".xml", modelFilePath, modelFileName, settingsFileName, descriptionFileName);

    //The following will load and compile and simulate the sbml model in the file
    simulation.SetCaseNumber(caseNumber);
    simulation.SetModelFilePath(modelFilePath);
    simulation.SetModelFileName(modelFileName);
    simulation.ReCompileIfDllExists(true);
    simulation.CopyFilesToOutputFolder();

    rr.setConservedMoietyAnalysis(false);

    if (!simulation.LoadSBMLFromFile())
    {
        Log(Logger::LOG_ERROR) << "Failed loading SBML model";
        throw("Failed loading SBML model");
    }
    //Then read settings file if it exists..
    string settingsOveride("");
    if (!simulation.LoadSettings(settingsOveride))
    {
        Log(Logger::LOG_ERROR) << "Failed loading SBML model settings";
        throw("Failed loading SBML model settings");
    }


    if (!isFBCTest(modelFilePath + "/" + descriptionFileName)) {
        //Only try simulating non-FBC tests.
        if (!simulation.Simulate())
        {
            Log(Logger::LOG_ERROR) << "Failed running simulation";
            throw("Failed running simulation");
        }
    }

}
